use std::arch::x86_64::_mm_sha256msg1_epu32;
use std::collections::VecDeque;
use std::fs::File;
use std::future::Future;
use std::io::Read;
use std::ops::Deref;
use std::path::PathBuf;
use std::str::Chars;
use std::sync::mpsc;
use std::sync::mpsc::{Receiver, Sender, SyncSender};
use anyhow::Context;
use crc32fast::hash;
use fast_image_resize::{IntoImageView, PixelType, ResizeOptions};
use ndarray::{arr1, array, s, Array, Array3, Array4, Ix, IxDyn};
use ort::execution_providers::OpenVINOExecutionProvider;
use ort::session::builder::{GraphOptimizationLevel, SessionBuilder};
use ort::session::Session;
use ort::value::Tensor;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use sha2::{Digest, Sha256};
use sha2::digest::generic_array::functional::FunctionalSequence;
use turbojpeg::{Decompressor, Image, PixelFormat};
use ameba_blog_downloader::data_dir;
use fast_image_resize::images::Image as fir_Image;
use futures::executor::block_on;
use futures::future::join_all;
use futures::TryFuture;
use num_traits::ToPrimitive;
use ort::inputs;
use tokio::{join, try_join};
use std::pin::Pin;
use kdam::{tqdm, BarExt};
use ort::tensor::Utf8Data;

static BATCH_SIZE: usize = 256;
static DECODE_FORMAT: PixelFormat = PixelFormat::RGB;
static INFERENCE_SIZE: usize = 112;
static FACE_FEATURE_MODEL: &[u8] = include_bytes!(r"C:\Users\tomokazu\PycharmProjects\RetinaFace_ONNX_Export\onnx_dest\arcface_unpg_f16_with_fp32_io.onnx");

async fn inference(receiver: Receiver<(Tensor<f32>, Vec<(PathBuf, String)>)>, file_count: usize) {
    ort::init().commit().unwrap();
    let model_hash = Sha256::digest(FACE_FEATURE_MODEL).to_ascii_lowercase().into_iter().map(|v| format!("{:02X}", v)).collect::<String>();
    let model = Session::builder().unwrap()
        .with_execution_providers([
            OpenVINOExecutionProvider::default().with_device_type("GPU").build().error_on_failure()
        ]).unwrap()
        .with_optimization_level(GraphOptimizationLevel::Level3).unwrap()
        .commit_from_memory(FACE_FEATURE_MODEL).unwrap();
    let mut futures = VecDeque::new();
    let mut bar = tqdm!(total=file_count,disable=false);
    let infer = async |(tensor, paths)| {
        let resp = model.run_async(inputs! {"input"=>tensor}.unwrap()).unwrap().await.unwrap();
        let out_tensor = resp.get("output").unwrap().try_extract_tensor::<f32>().unwrap().view().into_owned();
        // sender.send((out_tensor, paths)).unwrap();
        (out_tensor, paths)
    };
    let face_embeddings = hdf5_metno::File::append(data_dir().join("face_embeddings.hdf5")).unwrap();
    let save_emb = |emb: (Array<f32, IxDyn>, Vec<(PathBuf, String)>)| {
        for (order, (path, hash)) in emb.1.iter().enumerate() {
            let store_path = face_embeddings.group(model_hash.as_str()).unwrap().group(path.parent().unwrap().file_name().unwrap().to_str().unwrap()).unwrap();
            let builder = store_path.new_dataset_builder();
            if store_path.link_exists(path.file_name().unwrap().to_str().unwrap()) {
                store_path.unlink(path.file_name().unwrap().to_str().unwrap()).unwrap();
            }
            let dataset = builder.with_data(emb.0.slice(s![order,..])).create(path.file_name().unwrap().to_str().unwrap()).unwrap();
            let hash_vec = hash.as_utf8_bytes();
            let attr = dataset.new_attr::<u8>().shape(hash_vec.len()).create("hash").unwrap();
            attr.write(hash_vec).unwrap();
        }
        let update_len = emb.1.len();
        let member_name = emb.1.last().unwrap().clone().0.parent().unwrap().file_name().unwrap().to_str().unwrap().to_string();
        (update_len, member_name)
    };
    while match receiver.try_recv() {
        Ok(t) => {
            if t.1.is_empty() { false } else {
                futures.push_back(infer(t));
                true
            }
        }
        Err(_) => {
            while match futures.pop_front() {
                None => { false }
                Some(x) => {
                    let (len, postfix) = save_emb(x.await);

                    let postfix = postfix.to_owned() + "\0".repeat(postfix.chars().count()).as_str();
                    if bar.postfix != postfix {
                        bar.set_postfix(postfix);
                    }
                    bar.update(len).unwrap();
                    true
                }
            } {}
            true
        }
    } {}
    for future in futures {
        save_emb(future.await);
    }
    face_embeddings.close().unwrap();
    println!("fin inference");
    // sender.send((Array::default([0]).into_dyn(), vec![])).unwrap();
}

fn main() {
    // let model_hash = std::str::from_utf8(model_hash.as_slice()).unwrap();
    let (decode_sender, inference_receiver) = mpsc::sync_channel(100);

    // return;
    let model_hash = Sha256::digest(FACE_FEATURE_MODEL).to_ascii_lowercase().into_iter().map(|v| format!("{:02X}", v)).collect::<String>();
    let face_embeddings_cache = hdf5_metno::File::append(data_dir().join("face_embeddings.hdf5")).unwrap();

    let mut all_files = vec![];
    let model_cache_group = match face_embeddings_cache.group(model_hash.as_str()) {
        Ok(cache_group) => { cache_group }
        Err(_) => { face_embeddings_cache.create_group(model_hash.as_str()).unwrap() }
    };
    for member_dir in data_dir().join("face_cropped").read_dir().unwrap() {
        let member_dir = member_dir.unwrap();
        match model_cache_group.group(member_dir.file_name().clone().to_str().unwrap()) {
            Ok(member_cache_group) => { member_cache_group }
            Err(_) => { model_cache_group.create_group(member_dir.file_name().clone().to_str().unwrap()).unwrap() }
        };
        for image_file in member_dir.path().read_dir().unwrap() {
            let file_path = image_file.unwrap();
            let path = file_path.path();
            all_files.push(path);
        }
        // if member_dir.file_name().to_str().unwrap() == "上國料萌衣" { break; };
    }
    let all_files = all_files.into_par_iter().map(|file_name| {
        let mut file_buf = vec![];
        File::open(file_name.clone()).unwrap().read_to_end(&mut file_buf).unwrap();
        (Sha256::digest(file_buf).to_ascii_lowercase().into_iter().map(|v| format!("{:02X}", v)).collect::<String>(), file_name)
    }).collect::<Vec<_>>();
    let all_files = all_files.iter().filter_map(|(hash, path)| {
        match face_embeddings_cache.group(model_hash.as_str()).unwrap().
            group(path.clone().parent().unwrap().file_name().unwrap().to_str().unwrap()).unwrap()
            .dataset(path.file_name().clone().unwrap().to_str().unwrap()) {
            Ok(dataset) => {
                match dataset.attr("hash") {
                    Ok(data_hash) => {
                        let stored_hash = String::from_utf8(data_hash.read_raw::<u8>().unwrap()).unwrap();
                        if stored_hash == hash.clone() {
                            return None;
                        }
                    }
                    Err(_) => {}
                }
            }
            Err(_) => {}
        }
        Some((path.clone(), hash.clone()))
    }).collect::<Vec<_>>();
    face_embeddings_cache.close().unwrap();
    let file_len = all_files.len();
    let joiner = std::thread::spawn(move || {
        block_on(inference(inference_receiver, file_len))
    });
    // let thread_pool = ThreadPoolBuilder::new().num_threads(16).build().unwrap();
    let _ = all_files.chunks(BATCH_SIZE).collect::<Vec<_>>().into_par_iter().map(|file_names| {
        let mut decompressor = Decompressor::new().unwrap();
        let mut resizer = fast_image_resize::Resizer::new();
        unsafe { resizer.set_cpu_extensions(fast_image_resize::CpuExtensions::Avx2); }
        let mut tensor = Array4::zeros([file_names.len(), 3, INFERENCE_SIZE, INFERENCE_SIZE]);

        for (order, file) in file_names.iter().enumerate() {
            let mut fp = File::open(file.clone().0).unwrap();
            let mut bin = Vec::with_capacity(file.clone().0.metadata().unwrap().len() as usize);
            fp.read_to_end(&mut bin).unwrap();
            let header = decompressor.read_header(&bin).unwrap();
            let mut decoded = Image {
                pixels: vec![0; header.height * header.width * DECODE_FORMAT.size()],
                width: header.width,
                pitch: header.width * DECODE_FORMAT.size(),
                height: header.height,
                format: DECODE_FORMAT,
            };

            decompressor.decompress(&bin, decoded.as_deref_mut()).unwrap();
            let decoded = fir_Image::from_vec_u8(
                decoded.width as u32, decoded.height as u32, decoded.pixels, PixelType::U8x3,
            ).unwrap();
            let resize_scale = INFERENCE_SIZE as f64 / u32::max(decoded.width(), decoded.height()) as f64;

            let mut resized_image = fir_Image::new((resize_scale * decoded.width() as f64).round() as u32,
                                                   (resize_scale * decoded.height() as f64).round() as u32,
                                                   PixelType::U8x3);
            resizer.resize(&decoded, &mut resized_image, &ResizeOptions::default()).unwrap();
            // raw_images.push(resized_image);
            let pad_x = (INFERENCE_SIZE - resized_image.width() as usize).div_ceil(2);
            let pad_y = (INFERENCE_SIZE - resized_image.height() as usize).div_ceil(2);
            let image_tensor = Array3::from_shape_vec([resized_image.height() as usize, resized_image.width() as usize, 3usize], resized_image.buffer().to_vec()).unwrap().mapv(|v| v as f32 / 225.0);
            tensor.slice_mut(s![order,..,pad_y..pad_y+resized_image.height() as usize,pad_x..pad_x+resized_image.width() as usize]).assign(&image_tensor.permuted_axes([2, 0, 1]));
        }
        let tensor = Tensor::from_array(tensor).unwrap();
        // println!("decode_sender.send");
        decode_sender.send((tensor, file_names.to_vec())).unwrap();
    }).collect::<Vec<_>>();
    decode_sender.send((Tensor::from_array(array![[[[0.0]]]]).unwrap(), vec![])).unwrap();
    joiner.join().unwrap();
}