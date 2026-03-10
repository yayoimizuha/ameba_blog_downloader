use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::thread;

use fast_image_resize::images::Image as FirImage;
use fast_image_resize::{PixelType, ResizeOptions};
use futures::executor::block_on;
use image::imageops::crop_imm;
use image::{Rgb, RgbImage};
use imageproc::geometric_transformations::{rotate, Interpolation};
use itertools::Itertools;
use kdam::{tqdm, BarExt};
use ndarray::{arr1, arr2, Array, Array4, IxDyn};
use num_traits::FloatConst;
use ort::ep::{OpenVINOExecutionProvider, TensorRTExecutionProvider};
use ort::inputs;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::{RunOptions, Session};
use ort::value::{Tensor, Value};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tracing::debug;
use turbojpeg::{Decompressor, Image, PixelFormat};
use twox_hash::xxhash3_128::Hasher as XxHash3_128;

use ameba_blog_downloader::retinaface::found_face::FoundFace;
use ameba_blog_downloader::retinaface::retinaface_common::{ModelKind, RetinaFaceFaceDetector};
use ameba_blog_downloader::{data_dir, project_dir};

#[allow(unused_imports)]
use imageproc::drawing::draw_hollow_polygon_mut;
#[allow(unused_imports)]
use imageproc::point::Point;
use once_cell::sync::Lazy;

const BATCH_SIZE: usize = 32;
const LARGE_BATCH_SIZE: usize = BATCH_SIZE * 32;
const DECODE_FORMAT: PixelFormat = PixelFormat::RGB;
const INFERENCE_SIZE: usize = 640;

const MODEL_PATH: Lazy<PathBuf> = Lazy::new(|| project_dir().join("retinaface_resnet_fused_fp16_with_fp32_io.onnx"));

// -- Type aliases for channel payloads --

type InferenceBatch = (Tensor<f32>, Vec<PathBuf>, Vec<u128>);
type PostprocessBatch = (Array<f32, IxDyn>, Array<f32, IxDyn>, Array<f32, IxDyn>, Vec<usize>, Vec<PathBuf>);
type OriginalImage = (PathBuf, RgbImage, f64);

// -- Cache --

#[derive(Serialize, Deserialize)]
struct CachedInferenceResult {
    confidence: Vec<f32>,
    confidence_shape: Vec<usize>,
    loc: Vec<f32>,
    loc_shape: Vec<usize>,
    landmark: Vec<f32>,
    landmark_shape: Vec<usize>,
}

fn compute_model_hash(model_path: &str) -> String {
    let mut f = File::open(model_path).expect("Failed to open model file");
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 65536];
    loop {
        let n = f.read(&mut buf).expect("Failed to read model file");
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    format!("{:x}", hasher.finalize())
}

fn compute_image_hash(jpeg_bytes: &[u8]) -> u128 {
    XxHash3_128::oneshot(jpeg_bytes)
}

fn cache_path(cache_dir: &Path, model_hash: &str, image_hash: u128) -> PathBuf {
    cache_dir.join(format!("{}-{:032x}.bin", &model_hash[..16], image_hash))
}

fn load_cache(path: &Path) -> Option<CachedInferenceResult> {
    let data = fs::read(path).ok()?;
    bincode::serde::decode_from_slice::<CachedInferenceResult, _>(&data, bincode::config::standard())
        .ok()
        .map(|(v, _)| v)
}

fn save_cache(path: &Path, result: &CachedInferenceResult) {
    if let Ok(encoded) = bincode::serde::encode_to_vec(result, bincode::config::standard()) {
        let _ = fs::write(path, encoded);
    }
}

/// Concatenate a single field from cached results into one ndarray.
fn concat_cached_field(
    results: &[CachedInferenceResult],
    get_data: impl Fn(&CachedInferenceResult) -> &Vec<f32>,
    get_shape: impl Fn(&CachedInferenceResult) -> &Vec<usize>,
) -> Array<f32, IxDyn> {
    let arrays: Vec<_> = results
        .iter()
        .map(|r| {
            let mut shape = vec![1];
            shape.extend_from_slice(get_shape(r));
            Array::from_shape_vec(shape, get_data(r).clone()).unwrap()
        })
        .collect();
    let views: Vec<_> = arrays.iter().map(|a| a.view()).collect();
    ndarray::concatenate(ndarray::Axis(0), &views).unwrap()
}

fn reconstruct_from_cache(
    results: &[CachedInferenceResult],
) -> (Array<f32, IxDyn>, Array<f32, IxDyn>, Array<f32, IxDyn>) {
    (
        concat_cached_field(results, |r| &r.confidence, |r| &r.confidence_shape),
        concat_cached_field(results, |r| &r.loc, |r| &r.loc_shape),
        concat_cached_field(results, |r| &r.landmark, |r| &r.landmark_shape),
    )
}

// -- Inference thread --

async fn inference(
    receiver: Receiver<InferenceBatch>,
    sender: SyncSender<PostprocessBatch>,
    model_hash: String,
    cache_dir: PathBuf,
) {
    ameba_blog_downloader::init_ort();
    let mut model = Session::builder()
        .unwrap()
        .with_execution_providers([
            TensorRTExecutionProvider::default()
                .with_engine_cache(true)
                .with_fp16(true)
                .with_engine_cache_path("trt_cache")
                .build()
                .error_on_failure(),
            // OpenVINOExecutionProvider::default()
            //     .with_device_type("GPU")
            //     .with_precision("FP16")
            //     .build()
            //     .error_on_failure(),
        ])
        .unwrap()
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .unwrap()
        .commit_from_file(MODEL_PATH.as_path())
        .unwrap();

    let extract = |v: &Value| v.try_extract_array::<f32>().unwrap().to_owned();
    let opt = RunOptions::new().unwrap();

    while let Ok((tensor, paths, image_hashes)) = receiver.recv() {
        if paths.is_empty() {
            break;
        }

        let shape: Vec<usize> = tensor.shape().iter().map(|&v| v as usize).collect();

        let cached: Vec<Option<CachedInferenceResult>> = image_hashes
            .iter()
            .map(|&h| load_cache(&cache_path(&cache_dir, &model_hash, h)))
            .collect();

        let all_hit = cached.iter().all(|c| c.is_some());

        let (confidence, loc, landmark) = if all_hit {
            debug!("Batch fully cached ({} images), skipping inference.", paths.len());
            let results: Vec<_> = cached.into_iter().map(|c| c.unwrap()).collect();
            reconstruct_from_cache(&results)
        } else {
            let out = model
                .run_async(inputs! {"input" => tensor}, &opt)
                .unwrap()
                .await
                .unwrap();
            let confidence = extract(out.get("confidence").unwrap());
            let loc = extract(out.get("bbox").unwrap());
            let landmark = extract(out.get("landmark").unwrap());

            // Save cache for misses
            for (i, (hash, hit)) in image_hashes.iter().zip(cached.iter()).enumerate() {
                if hit.is_some() {
                    continue;
                }
                let entry = CachedInferenceResult {
                    confidence: confidence.index_axis(ndarray::Axis(0), i).iter().copied().collect(),
                    confidence_shape: confidence.shape()[1..].to_vec(),
                    loc: loc.index_axis(ndarray::Axis(0), i).iter().copied().collect(),
                    loc_shape: loc.shape()[1..].to_vec(),
                    landmark: landmark.index_axis(ndarray::Axis(0), i).iter().copied().collect(),
                    landmark_shape: landmark.shape()[1..].to_vec(),
                };
                save_cache(&cache_path(&cache_dir, &model_hash, *hash), &entry);
            }

            (confidence, loc, landmark)
        };

        if sender.send((confidence, loc, landmark, shape, paths)).is_err() {
            break;
        }
    }
}

// -- Geometry & cropping --

fn calc_tilt(landmarks: [[f32; 2]; 5]) -> f32 {
    let eye_center = [
        (landmarks[0][0] + landmarks[1][0]) / 2.0,
        (landmarks[0][1] + landmarks[1][1]) / 2.0,
    ];
    let mouth_center = [
        (landmarks[3][0] + landmarks[4][0]) / 2.0,
        (landmarks[3][1] + landmarks[4][1]) / 2.0,
    ];
    f32::atan2(eye_center[1] - mouth_center[1], eye_center[0] - mouth_center[0])
}

#[allow(dead_code)]
fn draw_rect(mut image: RgbImage, scale: f32, faces: &[FoundFace]) -> RgbImage {
    let inv = 1.0 / scale;
    for face in faces {
        let angle = calc_tilt(face.landmarks) + f32::PI() / 2.0;
        let cx = (face.bbox[0] + face.bbox[2]) * inv / 2.0;
        let cy = (face.bbox[1] + face.bbox[3]) * inv / 2.0;
        let rot = arr2(&[[angle.cos(), -angle.sin()], [angle.sin(), angle.cos()]]);
        let center = arr1(&[cx, cy]);
        let corners = [
            arr1(&[face.bbox[0] * inv, face.bbox[1] * inv]),
            arr1(&[face.bbox[2] * inv, face.bbox[1] * inv]),
            arr1(&[face.bbox[2] * inv, face.bbox[3] * inv]),
            arr1(&[face.bbox[0] * inv, face.bbox[3] * inv]),
        ];
        let pts: Vec<_> = corners
            .iter()
            .map(|c| {
                let r = rot.dot(&(c - &center)) + &center;
                Point::new(r[0], r[1])
            })
            .collect();
        draw_hollow_polygon_mut(&mut image, &pts, Rgb([255, 0, 0]));
    }
    image
}

fn crop_faces(image: &RgbImage, scale: f32, faces: &[FoundFace]) -> Vec<RgbImage> {
    let inv = 1.0 / scale;
    faces
        .iter()
        .filter(|f| f32::max(f.bbox[2] - f.bbox[0], f.bbox[3] - f.bbox[1]) * inv >= 100.0)
        .map(|face| {
            let angle = calc_tilt(face.landmarks) + f32::PI() / 2.0;
            let cx = (face.bbox[0] + face.bbox[2]) * inv / 2.0;
            let cy = (face.bbox[1] + face.bbox[3]) * inv / 2.0;
            let rotated = rotate(image, (cx, cy), -angle, Interpolation::Bilinear, Rgb([0, 0, 0]));
            let size = (f32::max(
                (face.bbox[2] - face.bbox[0]) * inv,
                (face.bbox[3] - face.bbox[1]) * inv,
            ) * 1.2) as u32;
            let half = size as f32 / 2.0;
            crop_imm(&rotated, (cx - half) as u32, (cy - half) as u32, size, size).to_image()
        })
        .collect()
}

// -- Postprocess thread --

fn postprocess(
    model_rx: Receiver<PostprocessBatch>,
    original_rx: Receiver<OriginalImage>,
    file_count: usize,
) {
    let detector = RetinaFaceFaceDetector {
        session: Session::builder().unwrap().commit_from_file(MODEL_PATH.as_path()).unwrap(),
        model: ModelKind::ResNet,
    };
    let mut originals: HashMap<PathBuf, (RgbImage, f64)> = HashMap::new();
    let mut bar = tqdm!(total = file_count, disable = false);
    let pool = rayon::ThreadPoolBuilder::new().num_threads(16).build().unwrap();
    let export_base = data_dir().join("face_cropped");

    for (confidence, loc, landmark, input_shape, paths) in model_rx.iter() {
        if input_shape.is_empty() {
            return;
        }

        // Drain available originals
        while let Ok((path, image, scale)) = original_rx.try_recv() {
            originals.insert(path, (image, scale));
        }

        let faces_vec = detector.post_process(confidence, loc, landmark, input_shape).unwrap();

        // Ensure export directories exist
        for dir in paths.iter().map(|p| p.parent().unwrap().file_name().unwrap()).unique() {
            let dest = export_base.join(dir);
            if !dest.exists() {
                fs::create_dir_all(&dest).unwrap();
            }
        }

        // Crop and save in parallel
        let work: Vec<_> = faces_vec
            .iter()
            .zip(&paths)
            .map(|(faces, path)| {
                let (image, scale) = &originals[path];
                (image.clone(), *scale as f32, faces.clone(), path.clone())
            })
            .collect();

        pool.install(|| {
            work.into_par_iter().for_each(|(image, scale, faces, path)| {
                let crops = crop_faces(&image, scale, &faces);
                let dest = export_base
                    .join(path.parent().unwrap().file_name().unwrap())
                    .join(path.file_name().unwrap());
                let stem = dest.to_str().unwrap().rsplitn(2, '.').nth(1).unwrap();
                for (i, crop) in crops.iter().enumerate() {
                    crop.save(format!("{stem}-{i:>02}.jpg")).unwrap();
                }
            });
        });

        // Clean up originals & update progress
        for path in &paths {
            originals.remove(path);
        }
        for (faces, path) in faces_vec.iter().zip(&paths) {
            debug!("{}", path.display());
            let mut postfix: String = path.parent().unwrap().file_name().unwrap().to_str().unwrap().into();
            postfix += &"\0".repeat(postfix.chars().count());
            if bar.postfix != postfix {
                bar.set_postfix(postfix);
            }
            bar.update(1).unwrap();
            for face in faces {
                debug!("\t{:?}", face);
            }
        }
    }
}

// -- Decode helpers --

struct DecodedImage {
    resized: FirImage<'static>,
    original: RgbImage,
    scale: f64,
}

fn decode_and_resize(
    bin: &[u8],
    decompressor: &mut Decompressor,
    resizer: &mut fast_image_resize::Resizer,
) -> DecodedImage {
    let header = decompressor.read_header(bin).unwrap();
    let mut decoded = Image {
        pixels: vec![0; header.height * header.width * DECODE_FORMAT.size()],
        width: header.width,
        pitch: header.width * DECODE_FORMAT.size(),
        height: header.height,
        format: DECODE_FORMAT,
    };
    decompressor.decompress(bin, decoded.as_deref_mut()).unwrap();

    let fir = FirImage::from_vec_u8(
        decoded.width as u32,
        decoded.height as u32,
        decoded.pixels,
        PixelType::U8x3,
    )
        .unwrap();

    let scale = INFERENCE_SIZE as f64 / u32::max(fir.width(), fir.height()) as f64;
    let resized = if scale >= 1.0 {
        fir.copy()
    } else {
        let w = (scale * fir.width() as f64).round() as u32;
        let h = (scale * fir.height() as f64).round() as u32;
        let mut dst = FirImage::new(w, h, PixelType::U8x3);
        resizer.resize(&fir, &mut dst, &ResizeOptions::default()).unwrap();
        dst
    };

    let original = RgbImage::from_raw(fir.width(), fir.height(), fir.into_vec()).unwrap();
    DecodedImage {
        resized,
        original,
        scale: f64::min(1.0, scale),
    }
}

fn build_padded_tensor(images: &[FirImage]) -> Array4<f32> {
    let n = images.len();
    let mut tensor = vec![0.0f32; n * INFERENCE_SIZE * INFERENCE_SIZE * 3];

    for (i, img) in images.iter().enumerate() {
        let w = img.width() as usize;
        let h = img.height() as usize;
        let buf = img.buffer();
        let base = i * INFERENCE_SIZE * INFERENCE_SIZE * 3;
        for y in 0..h {
            let dst_start = base + y * INFERENCE_SIZE * 3;
            let src_start = y * w * 3;
            for (dst, &src) in tensor[dst_start..dst_start + w * 3]
                .iter_mut()
                .zip(&buf[src_start..src_start + w * 3])
            {
                *dst = src as f32 / 255.0;
            }
        }
    }

    Array4::from_shape_vec((n, INFERENCE_SIZE, INFERENCE_SIZE, 3), tensor)
        .unwrap()
        .permuted_axes([0, 3, 1, 2])
}

// -- Main --

fn main() {
    tracing_subscriber::fmt::init();
    ameba_blog_downloader::init_ort();

    let model_hash = compute_model_hash(MODEL_PATH.as_path().to_str().unwrap());
    debug!("Model hash prefix: {}", &model_hash[..16]);

    let cache_dir = data_dir().join("inference_cache");
    fs::create_dir_all(&cache_dir).expect("Failed to create cache directory");

    let mut all_files = Vec::new();
    for entry in data_dir().join("blog_images").read_dir().unwrap() {
        for file in entry.unwrap().path().read_dir().unwrap() {
            all_files.push(file.unwrap().path());
        }
    }

    let (decode_tx, infer_rx) = mpsc::sync_channel::<InferenceBatch>(40);
    let (infer_tx, post_rx) = mpsc::sync_channel::<PostprocessBatch>(40);
    let (orig_tx, orig_rx) = mpsc::channel::<OriginalImage>();

    let mh = model_hash.clone();
    let cd = cache_dir.clone();
    let inference_handle = thread::spawn(move || block_on(inference(infer_rx, infer_tx, mh, cd)));

    let file_count = all_files.len();
    let postprocess_handle = thread::spawn(move || postprocess(post_rx, orig_rx, file_count));

    all_files
        .chunks(LARGE_BATCH_SIZE)
        .collect::<Vec<_>>()
        .into_par_iter()
        .for_each(|large_chunk| {
            let mut decompressor = Decompressor::new().unwrap();
            let mut resizer = fast_image_resize::Resizer::new();
            unsafe { resizer.set_cpu_extensions(fast_image_resize::CpuExtensions::Avx2); }

            for files in large_chunk.chunks(BATCH_SIZE) {
                let mut resized_images = Vec::with_capacity(files.len());
                let mut image_hashes = Vec::with_capacity(files.len());

                for file in files {
                    let bin = fs::read(file).unwrap();
                    image_hashes.push(compute_image_hash(&bin));

                    let decoded = decode_and_resize(&bin, &mut decompressor, &mut resizer);
                    orig_tx.send((file.clone(), decoded.original, decoded.scale)).unwrap();
                    resized_images.push(decoded.resized);
                }

                let input = build_padded_tensor(&resized_images);
                decode_tx
                    .send((Tensor::from_array(input).unwrap(), files.to_vec(), image_hashes))
                    .unwrap();
            }
            debug!("finished decode.");
        });

    // Termination signal
    decode_tx
        .send((
            Tensor::from_array(ndarray::array![[[[0.0]]]]).unwrap(),
            vec![],
            vec![],
        ))
        .unwrap();

    inference_handle.join().unwrap();
    postprocess_handle.join().unwrap();
}
