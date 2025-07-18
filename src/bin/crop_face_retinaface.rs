use std::collections::{HashMap, VecDeque};
use std::{fs, thread};
use std::cell::RefCell;
use std::fs::File;
use std::io::Read;
use std::ops::Deref;
use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::mpsc::{Receiver, SyncSender};
use fast_image_resize::{PixelType, ResizeOptions};
use rayon::prelude::*;
use ameba_blog_downloader::data_dir;
use turbojpeg::{Decompressor, Image, PixelFormat};
use fast_image_resize::images::Image as fir_Image;
use futures::executor::block_on;
use futures::future::{join_all, select_all};
use futures::FutureExt;
use futures::stream::SelectAll;
use ndarray::{arr1, arr2, array, Array, Array4, IxDyn};
use ort::session::{InferenceFut, NoSelectedOutputs, RunOptions, Session};
use ameba_blog_downloader::retinaface::retinaface_common::{ModelKind, RetinaFaceFaceDetector};
use image::{Rgb, RgbImage};
use image::imageops::{crop_imm};
use imageproc::drawing::{draw_hollow_polygon_mut};
use imageproc::geometric_transformations::{rotate, Interpolation};
use imageproc::point::Point;
use itertools::Itertools;
use kdam::{tqdm, BarExt};
use num_traits::{AsPrimitive, FloatConst};
use ort::execution_providers::OpenVINOExecutionProvider;
use ort::inputs;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::{Tensor, Value};
use tokio::pin;
use tokio_stream::StreamMap;
use tracing::debug;
use ameba_blog_downloader::retinaface::found_face::FoundFace;

static BATCH_SIZE: usize = 32;
static LARGE_BATCH_SIZE: usize = BATCH_SIZE * 32;
static DECODE_FORMAT: PixelFormat = PixelFormat::RGB;
static INFERENCE_SIZE: usize = 640;

static MODEL_PATH: &str = r"C:\Users\tomokazu\PycharmProjects\RetinaFace_ONNX_Export\onnx_dest\retinaface_resnet_fused_fp16_with_fp32_io.onnx";

// async fn inference(receiver: Receiver<(Tensor<f32>, Vec<PathBuf>)>, sender: SyncSender<(Array<f32, IxDyn>, Array<f32, IxDyn>, Array<f32, IxDyn>, Vec<usize>, Vec<PathBuf>)>) {
//     ort::init().commit().unwrap();
//     let mut model = Session::builder().unwrap()
//         .with_execution_providers([
//             OpenVINOExecutionProvider::default().with_device_type("GPU").build().error_on_failure()
//         ]).unwrap()
//         .with_optimization_level(GraphOptimizationLevel::Level3).unwrap()
//         .commit_from_file(MODEL_PATH).unwrap();
//     let futures = RefCell::new(VecDeque::new());
//     let extract_tensor = |tensor: &Value| tensor.try_extract_array::<f32>().unwrap().view().to_owned().mapv(|v| v);
//     let extract_output = async |model_output: InferenceFut, shape: Vec<i64>, path_vec: Vec<PathBuf>| {
//         let infer_out = model_output.await.unwrap();
//         let [ confidence, loc, landmark] = ["confidence", "bbox", "landmark"].map(|label| extract_tensor(infer_out.get(label).unwrap()));
//         sender.send((confidence, loc, landmark, shape.iter().map(|&v| v as usize).collect::<Vec<_>>(), path_vec)).unwrap();
//     };
//     let opt = RunOptions::new().unwrap();
// 
//     while if futures.borrow().len() > 15 {
//         let mut pop_vec = vec![];
//         for _ in 0..7 {
//             let (model_out, shape, path_vec) = futures.borrow_mut().pop_front().unwrap();
//             // extract_output(model_out, shape, path_vec).await;
//             pop_vec.push(extract_output(model_out, shape, path_vec));
//         }
//         join_all(pop_vec).await;
//         true
//     } else {
//         match receiver.try_recv() {
//             Ok((tensor, path_vec)) => {
//                 if path_vec.is_empty() { false } else {
//                     let tensor_shape = tensor.shape().clone();
//                     futures.borrow_mut().push_back((model.run_async(inputs! {"input"=>tensor}, &opt).unwrap(), tensor_shape.to_vec(), path_vec));
//                     true
//                 }
//             }
//             Err(_) => {
//                 while match futures.borrow_mut().pop_front() {
//                     None => { false }
//                     Some((model_out, shape, path_vec)) => {
//                         extract_output(model_out, shape, path_vec).await;
//                         true
//                     }
//                 } {}
//                 true
//             }
//         }
//     } {}
//     while match futures.borrow_mut().pop_front() {
//         None => { false }
//         Some((model_out, shape, path_vec)) => {
//             extract_output(model_out, shape, path_vec).await;
//             true
//         }
//     } {}
// }
async fn inference(receiver: Receiver<(Tensor<f32>, Vec<PathBuf>)>, sender: SyncSender<(Array<f32, IxDyn>, Array<f32, IxDyn>, Array<f32, IxDyn>, Vec<usize>, Vec<PathBuf>)>) {
    ort::init().commit().unwrap();
    let mut model = Session::builder().unwrap()
        .with_execution_providers([
            OpenVINOExecutionProvider::default().with_device_type("GPU").build().error_on_failure()
        ]).unwrap()
        .with_optimization_level(GraphOptimizationLevel::Level3).unwrap()
        .commit_from_file(MODEL_PATH).unwrap();

    let extract_tensor = |tensor: &Value| tensor.try_extract_array::<f32>().unwrap().to_owned();
    let opt = RunOptions::new().unwrap();

    // This loop processes one inference at a time. It waits to receive a tensor,
    // runs inference, and crucially `.await`s the result immediately. This releases the
    // mutable borrow on `model`, allowing the next loop iteration to borrow it again.
    while let Ok((tensor, path_vec)) = receiver.recv() {
        // The sending thread signals termination with an empty `path_vec`.
        if path_vec.is_empty() {
            break;
        }

        let shape = tensor.shape().iter().map(|&v| v as usize).collect::<Vec<_>>();

        // The core of the fix: call `run_async` and `await` it in the same expression.
        let infer_out = model.run_async(inputs! {"input" => tensor}, &opt)
            .unwrap()
            .await
            .unwrap();

        // Extract results from the completed inference.
        let [confidence, loc, landmark] = ["confidence", "bbox", "landmark"].map(|label| {
            extract_tensor(infer_out.get(label).unwrap())
        });

        // Send the results to the post-processing thread.
        if sender.send((confidence, loc, landmark, shape, path_vec)).is_err() {
            // The receiver has been dropped, so we can stop processing.
            break;
        }
    }
}
fn calc_tilt(landmarks: [[f32; 2]; 5]) -> f32 {
    let eye_center = [(landmarks[0][0] + landmarks[1][0]) / 2.0, (landmarks[0][1] + landmarks[1][1]) / 2.0];
    let mouse_center = [(landmarks[3][0] + landmarks[4][0]) / 2.0, (landmarks[3][1] + landmarks[4][1]) / 2.0];
    f32::atan2(eye_center[1] - mouse_center[1], eye_center[0] - mouse_center[0])
}

#[allow(dead_code)]
fn draw_rect(original_image: RgbImage, scale: f32, faces: Vec<FoundFace>) -> RgbImage {
    let scale = 1.0 / scale;
    let mut palette = original_image;
    for face in faces {
        let angle = calc_tilt(face.landmarks) + f32::PI() / 2.0;
        let center_x = (face.bbox[0] + face.bbox[2]) * scale / 2.0;
        let center_y = (face.bbox[1] + face.bbox[3]) * scale / 2.0;
        let rotation_matrix = arr2(&[
            [angle.cos(), -angle.sin()],
            [angle.sin(), angle.cos()],
        ]);
        let corners = [
            arr1(&[face.bbox[0] * scale, face.bbox[1] * scale]), // 左上
            arr1(&[face.bbox[2] * scale, face.bbox[1] * scale]), // 右上
            arr1(&[face.bbox[2] * scale, face.bbox[3] * scale]), // 右下
            arr1(&[face.bbox[0] * scale, face.bbox[3] * scale]), // 左下
        ];
        let rotated_corners = corners.map(|corner| {
            let relative_coords = corner - arr1(&[center_x, center_y]);
            let rotated_relative_coords = rotation_matrix.dot(&relative_coords);
            rotated_relative_coords + arr1(&[center_x, center_y])
        });
        draw_hollow_polygon_mut(&mut palette,
                                &[Point::new(rotated_corners[0][0], rotated_corners[0][1]),
                                    Point::new(rotated_corners[1][0], rotated_corners[1][1]),
                                    Point::new(rotated_corners[2][0], rotated_corners[2][1]),
                                    Point::new(rotated_corners[3][0], rotated_corners[3][1]),
                                ], Rgb::from([255, 0, 0]))
    }
    palette
}

fn crop_bbox(original_image: RgbImage, scale: f32, faces: Vec<FoundFace>) -> Vec<RgbImage> {
    let scale = 1.0 / scale;
    let mut crops = vec![];
    for face in faces {
        if f32::max(face.bbox[2] - face.bbox[0], face.bbox[3] - face.bbox[1]) * scale < 100.0 { continue; }
        let angle = calc_tilt(face.landmarks) + f32::PI() / 2.0;
        let center_x = (face.bbox[0] + face.bbox[2]) * scale / 2.0;
        let center_y = (face.bbox[1] + face.bbox[3]) * scale / 2.0;
        let rotated = rotate(&original_image, (center_x, center_y), -angle, Interpolation::Bilinear, Rgb([0, 0, 0]));
        let crop_size = (f32::max((face.bbox[2] - face.bbox[0]) * scale, (face.bbox[3] - face.bbox[1]) * scale) * 1.2) as u32;
        crops.push(crop_imm(&rotated, (center_x - crop_size as f32 / 2.0) as u32,
                            (center_y - crop_size as f32 / 2.0) as u32, crop_size, crop_size).to_image());
    }
    crops
}
fn postprocess(model_output_receiver: Receiver<(Array<f32, IxDyn>, Array<f32, IxDyn>, Array<f32, IxDyn>, Vec<usize>, Vec<PathBuf>)>, original_receiver: Receiver<(PathBuf, RgbImage, f64)>, file_count: usize) {
    let detector = RetinaFaceFaceDetector { session: Session::builder().unwrap().commit_from_file(MODEL_PATH).unwrap(), model: ModelKind::ResNet };
    let mut originals: HashMap<_, _> = HashMap::new();
    let mut bar = tqdm!(total=file_count,disable=false);
    let thread_pool = rayon::ThreadPoolBuilder::new().num_threads(16).build().unwrap();
    for (confidence, loc, landmark, input_shape, paths) in model_output_receiver.iter() {
        if input_shape.len() == 0 { return; }
        while match original_receiver.try_recv() {
            Ok((path, image, scale)) => {
                originals.insert(path.clone(), (image, scale));
                true
            }
            Err(_err) => { false }
        } {}

        let faces_vec = detector.post_process(confidence, loc, landmark, input_shape).unwrap();
        let member_dirs = paths.clone().into_iter().map(|path| path.parent().unwrap().file_name().unwrap().to_os_string()).unique().collect::<Vec<_>>();
        let export_base = data_dir().join("face_cropped");
        let _ = member_dirs.iter().map(|member_dir| if !export_base.join(member_dir).exists() { fs::create_dir_all(export_base.join(member_dir)).unwrap(); }).collect::<Vec<_>>();
        thread_pool.install(|| {
            let _ = faces_vec.clone().iter().zip(paths.clone()).map(|(faces, path)| {
                let (image, scale) = &originals[&path.clone()];
                (image, scale, faces, path.clone())
            }).collect::<Vec<_>>().into_par_iter().map(|(image, scale, faces, path)| {
                // let drawn_rect = draw_rect(image.clone(), scale.clone() as f32, faces.clone());
                // drawn_rect.save(export_base.join(path.parent().unwrap().file_name().unwrap()).join(path.file_name().unwrap())).unwrap();
                let crops = crop_bbox(image.clone(), scale.clone() as f32, faces.clone());
                let _ = crops.iter().enumerate().map(|(order, image)| {
                    let binding = export_base.join(path.parent().unwrap().file_name().unwrap()).join(path.file_name().unwrap());
                    let p = binding.to_str().unwrap().rsplitn(2, ".").nth(1).unwrap();
                    image.save(format!("{p}-{order:>02}.jpg")).unwrap();
                }).collect::<Vec<_>>();
            }).collect::<Vec<_>>();
        });

        let _ = paths.iter().map(|path| originals.remove(&path.clone())).collect::<Vec<_>>();
        let _ = faces_vec.iter().zip(paths).map(|(faces, path)| {
            debug!("{}", path.as_path().to_str().unwrap());
            let mut postfix: String = path.parent().unwrap().file_name().unwrap().to_str().unwrap().into();
            postfix = postfix.to_owned() + "\0".repeat(postfix.chars().count()).as_str();
            if bar.postfix != postfix {
                bar.set_postfix(postfix);
            }
            bar.update(1).unwrap();
            let _ = faces.iter().map(|face| {
                debug!("\t{:?}", face);
            }).collect::<Vec<_>>();
        }).collect::<Vec<_>>();
    }
}
fn main() {
    tracing_subscriber::fmt::init();
    ort::init().commit().unwrap();
    let mut all_files = vec![];
    for member_dir in data_dir().join("blog_images").read_dir().unwrap() {
        // for member_dir in PathBuf::from(r"C:\Users\tomokazu\すぐ消す\仮").read_dir().unwrap() {
        for image_file in member_dir.unwrap().path().read_dir().unwrap() {
            let path = image_file.unwrap().path();
            // println!("{:?}", path.clone());
            all_files.push(path);
        }
    }
    let (decoder_sender, inference_receiver) = mpsc::sync_channel(40);
    let (inference_sender, postprocess_receiver) = mpsc::sync_channel(40);
    let (original_sender, original_receiver) = mpsc::channel();
    let inference_handle = thread::spawn(move || {
        block_on(inference(inference_receiver, inference_sender));
    });
    let file_length = all_files.len().clone();
    let post_process_handle = thread::spawn(move || {
        postprocess(postprocess_receiver, original_receiver, file_length);
    });
    let _ = all_files.chunks(LARGE_BATCH_SIZE).collect::<Vec<_>>().into_par_iter().map(|large_chunk| {
        let _ = large_chunk.chunks(BATCH_SIZE).collect::<Vec<_>>().into_iter().map(|files| {
            let mut decompressor = Decompressor::new().unwrap();
            let mut resizer = fast_image_resize::Resizer::new();
            unsafe { resizer.set_cpu_extensions(fast_image_resize::CpuExtensions::Avx2); }
            let mut raw_images = Vec::new();
            for file in files {
                let mut fp = File::open(file).unwrap();
                let mut bin = Vec::with_capacity(file.metadata().unwrap().len() as usize);
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
                if resize_scale > 1.0 {
                    raw_images.push(decoded.copy());
                } else {
                    let mut resized_image = fir_Image::new((resize_scale * decoded.width() as f64).round() as u32,
                                                           (resize_scale * decoded.height() as f64).round() as u32,
                                                           PixelType::U8x3);
                    resizer.resize(&decoded, &mut resized_image, &ResizeOptions::default()).unwrap();
                    raw_images.push(resized_image);
                }

                original_sender.send((file.clone(), RgbImage::from_raw(header.width as u32, header.height as u32, decoded.into_vec()).unwrap(), f64::min(1.0, resize_scale))).unwrap();
            }
            let mut tensor = vec![0.0; raw_images.len() * INFERENCE_SIZE * INFERENCE_SIZE * 3];
            for i in 0..raw_images.len() {
                let float_buffer = raw_images[i].buffer().into_iter().map(|&v| v as f32 / 255.0).collect::<Vec<_>>();
                for j in 0..raw_images[i].height() as usize {
                    let begin = i * INFERENCE_SIZE * INFERENCE_SIZE * 3 + INFERENCE_SIZE * 3 * j;
                    let end = i * INFERENCE_SIZE * INFERENCE_SIZE * 3 + INFERENCE_SIZE * 3 * j + 3 * raw_images[i].width() as usize;
                    let src = &float_buffer[j * raw_images[i].width() as usize * 3..(j + 1) * raw_images[i].width() as usize * 3];
                    tensor[begin..end].copy_from_slice(src);
                }
            }
            let input_tensor = Array4::from_shape_vec((tensor.len() / (INFERENCE_SIZE * INFERENCE_SIZE * 3), INFERENCE_SIZE, INFERENCE_SIZE, 3), tensor.clone()).unwrap().permuted_axes([0, 3, 1, 2]);

            decoder_sender.send((Tensor::from_array(input_tensor).unwrap(), files.to_vec())).unwrap();
            // (tensor, raw_images.into_iter().map(|(_, a, b)| (a, b)).collect::<Vec<_>>())
        }).collect::<Vec<_>>();


        debug!("{}", "finished decode.");
        // println!("{}", "finished decode.");
    }).collect::<Vec<_>>();
    decoder_sender.send((Tensor::from_array(array![[[[0.0]]]]).unwrap(), vec![])).unwrap();
    inference_handle.join().unwrap();
    post_process_handle.join().unwrap();
}