use ameba_blog_downloader::{data_dir, Entities, Entity};
use bincode::config;
use fast_image_resize::images::Image as fir_Image;
use fast_image_resize::{PixelType, ResizeOptions};
use futures::future::join_all;
use kdam::{tqdm, BarExt};
use ndarray::{array, s, Array, Array3, Array4, IxDyn};
use ort::ep::{OpenVINOExecutionProvider, TensorRTExecutionProvider};
use ort::inputs;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::{RunOptions, Session};
use ort::value::Tensor;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs::File;
use std::io::{Read, Write};
use std::sync::mpsc;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;
use tokio::io::AsyncReadExt;
use tokio::sync::Semaphore;
use turbojpeg::{Decompressor, Image, PixelFormat};
use twox_hash::XxHash3_128;

static BATCH_SIZE: usize = 256;
static DECODE_FORMAT: PixelFormat = PixelFormat::RGB;
static INFERENCE_SIZE: usize = 112;
static FACE_FEATURE_MODEL: &[u8] = include_bytes!(r"../../arcface_unpg_f16_with_fp32_io.onnx");

async fn inference(receiver: Receiver<(Tensor<f32>, Vec<(String, u128)>)>, sender: Sender<(Array<f32, IxDyn>, Vec<(String, u128)>)>) {
    ameba_blog_downloader::init_ort();
    let shape_min = format!("input:1x3x{INFERENCE_SIZE}x{INFERENCE_SIZE}");
    let shape_opt = format!("input:{BATCH_SIZE}x3x{INFERENCE_SIZE}x{INFERENCE_SIZE}");
    let shape_max = format!("input:{BATCH_SIZE}x3x{INFERENCE_SIZE}x{INFERENCE_SIZE}");

    let mut model = Session::builder().unwrap()
        .with_execution_providers([
            // TensorRTExecutionProvider::default()
            //     .with_max_workspace_size(2 * 1024 * 1024 * 1024) // 2 GB
            //     .with_fp16(true)
            //     .with_engine_cache(true)
            //     .with_engine_cache_path("trt_cache")
            //     .with_timing_cache(true)
            //     .with_timing_cache_path("trt_cache")
            //     .with_build_heuristics(true)
            //     .with_context_memory_sharing(true)
            //     .with_profile_min_shapes(&shape_min)
            //     .with_profile_opt_shapes(&shape_opt)
            //     .with_profile_max_shapes(&shape_max)
            //     .build()
            //     .error_on_failure(),
            OpenVINOExecutionProvider::default()
                .with_device_type("GPU")
                .with_precision("FP16")
                .build().error_on_failure()
        ]).unwrap()
        .with_optimization_level(GraphOptimizationLevel::Level3).unwrap()
        .with_intra_threads(16).unwrap()
        .commit_from_memory(FACE_FEATURE_MODEL).unwrap();

    let opt = RunOptions::new().unwrap();

    // Use a blocking `for` loop on the receiver. This is more idiomatic and efficient
    // than a `try_recv` busy-loop. The task will sleep until a message is ready.
    for (tensor, metas) in receiver {
        // Check for the sentinel value (a batch with no metadata) to end the loop.
        if metas.is_empty() {
            break;
        }

        // Run one inference at a time. The `.await` will pause this loop,
        // yielding control until the GPU operation is complete.
        let resp = model.run_async(inputs! {"input"=>tensor}, &opt).unwrap().await.unwrap();
        let out_tensor = resp.get("output").unwrap().try_extract_array::<f32>().unwrap().view().into_owned();

        // Send the result to the collector. If it fails, the collector has shut down.
        if sender.send((out_tensor, metas)).is_err() {
            eprintln!("Collector thread disconnected, inference is stopping.");
            break;
        }
    }

    println!("fin inference");
    // Send the final sentinel value to signal the collector to finish.
    let _ = sender.send((Array::default([0]).into_dyn(), vec![]));
}
fn collector(receiver: Receiver<(Array<f32, IxDyn>, Vec<(String, u128)>)>, entities: Entities, proc_len: usize) -> Entities {
    let mut entities = entities;
    let mut bar = tqdm!(total=proc_len,disable=false);
    while match receiver.try_recv() {
        Ok((tensor, metas)) => {
            match metas.len() {
                0 => { false }
                _ => {
                    for (order, (path, hash)) in metas.into_iter().enumerate() {
                        bar.update(1).unwrap();
                        entities.0.insert(Entity { file_name: path, file_hash: hash, embeddings: tensor.slice(s![order,..]).to_vec() });
                    };
                    true
                }
            }
        }
        Err(_) => { true }
    } {}
    entities
}
#[tokio::main]
async fn main() {
    let model_hash = Sha256::digest(FACE_FEATURE_MODEL).to_ascii_lowercase().into_iter().map(|v| format!("{:02X}", v)).collect::<String>();
    let entities: Entities = if data_dir().join("embeddings_cache").join(format!("model_{model_hash}.bin")).exists() {
        let mut cache = Vec::new();
        File::open(data_dir().join("embeddings_cache").join(format!("model_{model_hash}.bin"))).unwrap().read_to_end(&mut cache).unwrap();
        bincode::decode_from_slice(cache.as_slice(), config::standard()).unwrap().0
    } else { Entities(HashSet::new()) };
    let sem = Arc::new(Semaphore::new(64));
    let mut all_files = Vec::new();
    for member_dir in data_dir().join("face_cropped").read_dir().unwrap() {
        let member_dir = member_dir.unwrap();
        let future = member_dir.path().read_dir().unwrap().map(|file_path| {
            let file_path = file_path.unwrap();
            let sem = sem.clone();
            tokio::spawn(async move {
                let _permit = sem.acquire().await.unwrap();
                let path = file_path.path();
                let mut buf = Vec::new();
                tokio::fs::File::open(&path).await.unwrap().read_to_end(&mut buf).await.unwrap();
                let hash = XxHash3_128::oneshot(buf.as_slice());
                (buf, Entity { file_name: path.to_str().unwrap().to_owned(), file_hash: hash, embeddings: vec![] })
            })
        }).collect::<Vec<_>>();
        join_all(future).await.into_iter().for_each(|v| {
            let (buf, entity) = v.unwrap();
            if !entities.0.contains(&entity) {
                all_files.push((entity.file_name, buf, entity.file_hash));
            }
        });
    }

    let (decode_sender, inference_receiver) = mpsc::sync_channel(100);
    let (inference_sender, collector_receiver) = mpsc::channel();


    let joiner_1 = tokio::spawn(inference(inference_receiver, inference_sender));
    let proc_len = all_files.len();
    let joiner_2 = std::thread::spawn(move || { collector(collector_receiver, entities, proc_len) });

    let _ = all_files.chunks(BATCH_SIZE).collect::<Vec<_>>().into_par_iter().for_each(|file_names| {
        let mut decompressor = Decompressor::new().unwrap();
        let mut resizer = fast_image_resize::Resizer::new();
        unsafe { resizer.set_cpu_extensions(fast_image_resize::CpuExtensions::Avx2); }
        let mut tensor = Array4::zeros([file_names.len(), 3, INFERENCE_SIZE, INFERENCE_SIZE]);

        for (order, (_path, buf, _hash)) in file_names.iter().enumerate() {
            let header = decompressor.read_header(&buf).unwrap();
            let mut decoded = Image {
                pixels: vec![0; header.height * header.width * DECODE_FORMAT.size()],
                width: header.width,
                pitch: header.width * DECODE_FORMAT.size(),
                height: header.height,
                format: DECODE_FORMAT,
            };

            decompressor.decompress(&buf, decoded.as_deref_mut()).unwrap();
            let decoded = fir_Image::from_vec_u8(
                decoded.width as u32, decoded.height as u32, decoded.pixels, PixelType::U8x3,
            ).unwrap();
            let resize_scale = INFERENCE_SIZE as f64 / u32::max(decoded.width(), decoded.height()) as f64;

            let mut resized_image = fir_Image::new((resize_scale * decoded.width() as f64).round() as u32,
                                                   (resize_scale * decoded.height() as f64).round() as u32,
                                                   PixelType::U8x3);
            resizer.resize(&decoded, &mut resized_image, &ResizeOptions::default()).unwrap();
            let pad_x = (INFERENCE_SIZE - resized_image.width() as usize).div_ceil(2);
            let pad_y = (INFERENCE_SIZE - resized_image.height() as usize).div_ceil(2);
            let image_tensor = Array3::from_shape_vec([resized_image.height() as usize, resized_image.width() as usize, 3usize], resized_image.buffer().to_vec()).unwrap().mapv(|v| v as f32 / 225.0);
            tensor.slice_mut(s![order,..,pad_y..pad_y+resized_image.height() as usize,pad_x..pad_x+resized_image.width() as usize]).assign(&image_tensor.permuted_axes([2, 0, 1]));
        }
        let tensor = Tensor::from_array(tensor).unwrap();
        decode_sender.send((tensor, file_names.iter().map(|(path, _, hash)| { (path.clone(), hash.clone()) }).collect::<Vec<_>>())).unwrap();
    });
    decode_sender.send((Tensor::from_array(array![[[[0.0]]]]).unwrap(), vec![])).unwrap();
    joiner_1.await.unwrap();
    File::create(data_dir().join("embeddings_cache").join(format!("model_{model_hash}.bin"))).unwrap().write_all({
        bincode::encode_to_vec(joiner_2.join().unwrap(), config::standard()).unwrap().as_slice()
    }).unwrap();

}
