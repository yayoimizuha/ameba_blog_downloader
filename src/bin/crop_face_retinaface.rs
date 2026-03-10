use std::collections::HashMap;
use std::{fs, thread};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::sync::mpsc::{Receiver, SyncSender};
use fast_image_resize::{PixelType, ResizeOptions};
use rayon::prelude::*;
use ameba_blog_downloader::data_dir;
use turbojpeg::{Decompressor, Image, PixelFormat};
use fast_image_resize::images::Image as fir_Image;
use futures::executor::block_on;
use ndarray::{arr1, arr2, Array, Array4, IxDyn};
use ort::session::{RunOptions, Session};
use ameba_blog_downloader::retinaface::retinaface_common::{ModelKind, RetinaFaceFaceDetector};
use image::{Rgb, RgbImage};
use image::imageops::crop_imm;
use imageproc::drawing::draw_hollow_polygon_mut;
use imageproc::geometric_transformations::{rotate, Interpolation};
use imageproc::point::Point;
use itertools::Itertools;
use kdam::{tqdm, BarExt};
use num_traits::FloatConst;
use ort::ep::{OpenVINOExecutionProvider, TensorRTExecutionProvider};
use ort::inputs;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::{Tensor, Value};
use tracing::debug;
use sha2::{Sha256, Digest};
use twox_hash::xxhash3_128::Hasher as XxHash3_128;
use serde::{Serialize, Deserialize};
use ameba_blog_downloader::retinaface::found_face::FoundFace;

static BATCH_SIZE: usize = 32;
static LARGE_BATCH_SIZE: usize = BATCH_SIZE * 32;
static DECODE_FORMAT: PixelFormat = PixelFormat::RGB;
static INFERENCE_SIZE: usize = 640;

static MODEL_PATH: &str = r"C:\Users\tomokazu\PycharmProjects\RetinaFace_ONNX_Export\onnx_dest\retinaface_resnet_fused_fp16_with_fp32_io.onnx";

// ---------------------------------------------------------------------------
// キャッシュ関連
// ---------------------------------------------------------------------------

/// 推論結果を保存するためのシリアライズ可能な構造体
#[derive(Serialize, Deserialize)]
struct CachedInferenceResult {
    confidence: Vec<f32>,
    confidence_shape: Vec<usize>,
    loc: Vec<f32>,
    loc_shape: Vec<usize>,
    landmark: Vec<f32>,
    landmark_shape: Vec<usize>,
}

/// モデルファイルの SHA-256 ハッシュを計算する（起動時に一度だけ呼ぶ）
fn compute_model_hash(model_path: &str) -> String {
    let mut f = File::open(model_path).expect("Failed to open model file for hashing");
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 65536];
    loop {
        let n = f.read(&mut buf).expect("Failed to read model file");
        if n == 0 { break; }
        hasher.update(&buf[..n]);
    }
    format!("{:x}", hasher.finalize())
}

/// JPEG バイナリの XXHash3-128 ハッシュを計算する（高速）
fn compute_image_hash(jpeg_bytes: &[u8]) -> u128 {
    XxHash3_128::oneshot(jpeg_bytes)
}

/// キャッシュファイルのパスを返す
///
/// ファイル名: `<model_hash_prefix16>-<image_hash32hex>.bin`
fn cache_path(cache_dir: &Path, model_hash: &str, image_hash: u128) -> PathBuf {
    cache_dir.join(format!("{}-{:032x}.bin", &model_hash[..16], image_hash))
}

/// キャッシュから推論結果を読み込む（存在しなければ None）
fn load_cache(path: &Path) -> Option<CachedInferenceResult> {
    let data = fs::read(path).ok()?;
    bincode::serde::decode_from_slice::<CachedInferenceResult, _>(
        &data,
        bincode::config::standard(),
    )
    .ok()
    .map(|(v, _)| v)
}

/// 推論結果をキャッシュに書き込む
fn save_cache(path: &Path, result: &CachedInferenceResult) {
    if let Ok(encoded) = bincode::serde::encode_to_vec(result, bincode::config::standard()) {
        let _ = fs::write(path, encoded);
    }
}

// ---------------------------------------------------------------------------
// 推論スレッド
// ---------------------------------------------------------------------------

/// デコーダスレッドから受け取るメッセージ
/// - `tensor`: バッチ入力テンソル
/// - `path_vec`: バッチ内の各画像のパス
/// - `image_hashes`: 各画像の XXHash3-128 ハッシュ（path_vec と同じ順序）
async fn inference(
    receiver: Receiver<(Tensor<f32>, Vec<PathBuf>, Vec<u128>)>,
    sender: SyncSender<(Array<f32, IxDyn>, Array<f32, IxDyn>, Array<f32, IxDyn>, Vec<usize>, Vec<PathBuf>)>,
    model_hash: String,
    cache_dir: PathBuf,
) {
    ameba_blog_downloader::init_ort();
    let mut model = Session::builder().unwrap()
        .with_execution_providers([
            TensorRTExecutionProvider::default().with_fp16(true).with_int8(true).build(),
            OpenVINOExecutionProvider::default().with_device_type("GPU").with_precision("FP16").build().error_on_failure()
        ]).unwrap()
        .with_optimization_level(GraphOptimizationLevel::Level3).unwrap()
        .commit_from_file(MODEL_PATH).unwrap();

    let extract_tensor = |tensor: &Value| tensor.try_extract_array::<f32>().unwrap().to_owned();
    let opt = RunOptions::new().unwrap();

    while let Ok((tensor, path_vec, image_hashes)) = receiver.recv() {
        // 終了シグナル: path_vec が空
        if path_vec.is_empty() {
            break;
        }

        let batch_size = path_vec.len();

        // ------------------------------------------------------------------
        // 1. 各画像のキャッシュ確認
        // ------------------------------------------------------------------
        let cached_results: Vec<Option<CachedInferenceResult>> = image_hashes
            .iter()
            .map(|&h| load_cache(&cache_path(&cache_dir, &model_hash, h)))
            .collect();

        let all_cached = cached_results.iter().all(|r| r.is_some());

        // ------------------------------------------------------------------
        // 2. キャッシュミスがあれば実際に推論を実行する
        // ------------------------------------------------------------------
        let (confidence_full, loc_full, landmark_full, shape) = if all_cached {
            debug!("All {} images in batch are cache hits, skipping inference.", batch_size);
            // ダミーとして使う shape だけ取得（後で上書き）
            let s = tensor.shape().iter().map(|&v| v as usize).collect::<Vec<_>>();
            (None, None, None, s)
        } else {
            let shape = tensor.shape().iter().map(|&v| v as usize).collect::<Vec<_>>();
            let infer_out = model.run_async(inputs! {"input" => tensor}, &opt)
                .unwrap()
                .await
                .unwrap();
            let [confidence, loc, landmark] = ["confidence", "bbox", "landmark"].map(|label| {
                extract_tensor(infer_out.get(label).unwrap())
            });
            (Some(confidence), Some(loc), Some(landmark), shape)
        };

        // ------------------------------------------------------------------
        // 3. 推論結果をキャッシュに書き込み、同時に送信用データを組み立てる
        // ------------------------------------------------------------------
        // バッチ全体の結果を再構成するための配列を用意する。
        // キャッシュヒット画像: キャッシュから復元
        // キャッシュミス画像: 推論結果の対応スライスから取り出す
        //
        // 推論結果の形状:
        //   confidence: [N, num_anchors, 2]
        //   loc:        [N, num_anchors, 4]
        //   landmark:   [N, num_anchors, 10]
        //
        // ここでは送信用に「バッチ全体の ndarray」を再合成する。

        if all_cached {
            // 全てキャッシュヒット: キャッシュから ndarray を復元して結合
            let results: Vec<CachedInferenceResult> = cached_results
                .into_iter()
                .map(|r| r.unwrap())
                .collect();

            let (confidence_arr, loc_arr, landmark_arr) =
                reconstruct_arrays_from_cache(&results);

            if sender.send((confidence_arr, loc_arr, landmark_arr, shape, path_vec)).is_err() {
                break;
            }
        } else {
            // 一部またはすべてキャッシュミス
            let confidence_full = confidence_full.unwrap();
            let loc_full = loc_full.unwrap();
            let landmark_full = landmark_full.unwrap();

            // キャッシュミス画像を個別にキャッシュ保存
            for (i, (hash, cached)) in image_hashes.iter().zip(cached_results.iter()).enumerate() {
                if cached.is_none() {
                    let conf_slice = confidence_full.index_axis(ndarray::Axis(0), i).to_owned();
                    let loc_slice = loc_full.index_axis(ndarray::Axis(0), i).to_owned();
                    let lm_slice = landmark_full.index_axis(ndarray::Axis(0), i).to_owned();

                    let entry = CachedInferenceResult {
                        confidence: conf_slice.iter().cloned().collect(),
                        confidence_shape: conf_slice.shape().to_vec(),
                        loc: loc_slice.iter().cloned().collect(),
                        loc_shape: loc_slice.shape().to_vec(),
                        landmark: lm_slice.iter().cloned().collect(),
                        landmark_shape: lm_slice.shape().to_vec(),
                    };
                    save_cache(&cache_path(&cache_dir, &model_hash, *hash), &entry);
                }
            }

            if sender.send((confidence_full, loc_full, landmark_full, shape, path_vec)).is_err() {
                break;
            }
        }
    }
}

/// キャッシュから復元した結果群を結合して ndarray にする
fn reconstruct_arrays_from_cache(
    results: &[CachedInferenceResult],
) -> (Array<f32, IxDyn>, Array<f32, IxDyn>, Array<f32, IxDyn>) {
    let conf_views: Vec<_> = results.iter().map(|r| {
        let mut shape = vec![1];
        shape.extend_from_slice(&r.confidence_shape);
        Array::from_shape_vec(shape, r.confidence.clone()).unwrap()
    }).collect();

    let loc_views: Vec<_> = results.iter().map(|r| {
        let mut shape = vec![1];
        shape.extend_from_slice(&r.loc_shape);
        Array::from_shape_vec(shape, r.loc.clone()).unwrap()
    }).collect();

    let lm_views: Vec<_> = results.iter().map(|r| {
        let mut shape = vec![1];
        shape.extend_from_slice(&r.landmark_shape);
        Array::from_shape_vec(shape, r.landmark.clone()).unwrap()
    }).collect();

    let conf_refs: Vec<_> = conf_views.iter().map(|a| a.view()).collect();
    let loc_refs: Vec<_> = loc_views.iter().map(|a| a.view()).collect();
    let lm_refs: Vec<_> = lm_views.iter().map(|a| a.view()).collect();

    (
        ndarray::concatenate(ndarray::Axis(0), &conf_refs).unwrap(),
        ndarray::concatenate(ndarray::Axis(0), &loc_refs).unwrap(),
        ndarray::concatenate(ndarray::Axis(0), &lm_refs).unwrap(),
    )
}

// ---------------------------------------------------------------------------
// 角度計算・描画・クロップ
// ---------------------------------------------------------------------------

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
            arr1(&[face.bbox[0] * scale, face.bbox[1] * scale]),
            arr1(&[face.bbox[2] * scale, face.bbox[1] * scale]),
            arr1(&[face.bbox[2] * scale, face.bbox[3] * scale]),
            arr1(&[face.bbox[0] * scale, face.bbox[3] * scale]),
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

// ---------------------------------------------------------------------------
// ポストプロセススレッド
// ---------------------------------------------------------------------------

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
                let crops = crop_bbox(image.clone(), *scale as f32, faces.clone());
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

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    tracing_subscriber::fmt::init();
    ameba_blog_downloader::init_ort();

    // モデルハッシュをメインスレッドで一度だけ計算する
    let model_hash = compute_model_hash(MODEL_PATH);
    debug!("Model hash (SHA-256 prefix): {}", &model_hash[..16]);

    // キャッシュディレクトリを作成する
    let cache_dir = data_dir().join("inference_cache");
    fs::create_dir_all(&cache_dir).expect("Failed to create inference cache directory");

    let mut all_files = vec![];
    for member_dir in data_dir().join("blog_images").read_dir().unwrap() {
        for image_file in member_dir.unwrap().path().read_dir().unwrap() {
            let path = image_file.unwrap().path();
            all_files.push(path);
        }
    }

    // チャンネル: デコーダ → 推論 (パスと画像ハッシュを追加)
    let (decoder_sender, inference_receiver) = mpsc::sync_channel::<(Tensor<f32>, Vec<PathBuf>, Vec<u128>)>(40);
    let (inference_sender, postprocess_receiver) = mpsc::sync_channel(40);
    let (original_sender, original_receiver) = mpsc::channel();

    let model_hash_clone = model_hash.clone();
    let cache_dir_clone = cache_dir.clone();
    let inference_handle = thread::spawn(move || {
        block_on(inference(inference_receiver, inference_sender, model_hash_clone, cache_dir_clone));
    });

    let file_length = all_files.len();
    let post_process_handle = thread::spawn(move || {
        postprocess(postprocess_receiver, original_receiver, file_length);
    });

    let _ = all_files.chunks(LARGE_BATCH_SIZE).collect::<Vec<_>>().into_par_iter().map(|large_chunk| {
        let _ = large_chunk.chunks(BATCH_SIZE).collect::<Vec<_>>().into_iter().map(|files| {
            let mut decompressor = Decompressor::new().unwrap();
            let mut resizer = fast_image_resize::Resizer::new();
            unsafe { resizer.set_cpu_extensions(fast_image_resize::CpuExtensions::Avx2); }
            let mut raw_images = Vec::new();
            let mut image_hashes = Vec::new();

            for file in files {
                let mut fp = File::open(file).unwrap();
                let mut bin = Vec::with_capacity(file.metadata().unwrap().len() as usize);
                fp.read_to_end(&mut bin).unwrap();

                // JPEG バイナリのハッシュをここで計算する
                image_hashes.push(compute_image_hash(&bin));

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
            let input_tensor = Array4::from_shape_vec(
                (tensor.len() / (INFERENCE_SIZE * INFERENCE_SIZE * 3), INFERENCE_SIZE, INFERENCE_SIZE, 3),
                tensor,
            ).unwrap().permuted_axes([0, 3, 1, 2]);

            // image_hashes も一緒に送信する
            decoder_sender.send((Tensor::from_array(input_tensor).unwrap(), files.to_vec(), image_hashes)).unwrap();
        }).collect::<Vec<_>>();

        debug!("{}", "finished decode.");
    }).collect::<Vec<_>>();

    // 終了シグナル: path_vec と image_hashes を空にして送る
    decoder_sender.send((
        Tensor::from_array(ndarray::array![[[[0.0]]]]).unwrap(),
        vec![],
        vec![],
    )).unwrap();

    inference_handle.join().unwrap();
    post_process_handle.join().unwrap();
}
