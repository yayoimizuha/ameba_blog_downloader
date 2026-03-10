// RetinaFace を用いた顔検出＆クロップバイナリ
//
// パイプライン構成:
//   decode スレッド群 (rayon)  →  inference スレッド  →  crop_and_export スレッド
//
// キャッシュ:
//   post_process 後の Vec<FoundFace> を SQLite ($DATA_PATH/face_cache.sqlite) に保存し、
//   同一画像は推論をスキップする。

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::thread;

use fast_image_resize::images::Image as FirImage;
use fast_image_resize::{PixelType, ResizeOptions};
use image::imageops::crop_imm;
use image::{Rgb, RgbImage};
use imageproc::geometric_transformations::{rotate, Interpolation};
use itertools::Itertools;
use kdam::{tqdm, BarExt};
use ndarray::Array4;
use num_traits::FloatConst;
use once_cell::sync::Lazy;
use ort::ep::TensorRTExecutionProvider;
use ort::inputs;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::{Tensor, Value};
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use sqlx::sqlite::SqliteConnectOptions;
use sqlx::SqlitePool;
use tracing::debug;
use turbojpeg::{Decompressor, Image, PixelFormat};
use twox_hash::xxhash3_128::Hasher as XxHash3_128;

use ameba_blog_downloader::retinaface::found_face::FoundFace;
use ameba_blog_downloader::retinaface::retinaface_common::{ModelKind, RetinaFaceFaceDetector};
use ameba_blog_downloader::{data_dir, project_dir};

// デバッグ用描画 (draw_rect) で使用
#[allow(unused_imports)]
use imageproc::drawing::draw_hollow_polygon_mut;
#[allow(unused_imports)]
use imageproc::point::Point;
#[allow(unused_imports)]
use ndarray::{arr1, arr2};

// ============================================================================
// 定数
// ============================================================================

const BATCH_SIZE: usize = 128;
const LARGE_BATCH_SIZE: usize = BATCH_SIZE * 32;
const DECODE_FORMAT: PixelFormat = PixelFormat::RGB;
const INFERENCE_SIZE: usize = 640;

static MODEL_PATH: Lazy<PathBuf> =
    Lazy::new(|| project_dir().join("retinaface_resnet_fused_fp16_with_fp32_io.onnx"));

// ============================================================================
// チャネルペイロード型
// ============================================================================

/// デコードスレッド → 推論スレッド: バッチテンソル、ファイルパス、画像ハッシュ
type InferenceBatch = (Tensor<f32>, Vec<PathBuf>, Vec<u128>);

/// 推論スレッド → クロップスレッド: 画像ごとの検出顔リスト、ファイルパス
type FaceDetectionBatch = (Vec<Vec<FoundFace>>, Vec<PathBuf>);

/// デコードスレッド → クロップスレッド: 元画像（クロップ用）
type OriginalImage = (PathBuf, RgbImage, f64);

// ============================================================================
// キャッシュ (SQLite)
// ============================================================================

/// モデルファイルの SHA-256 ハッシュを計算する
fn compute_model_hash(model_path: &str) -> String {
    use std::io::Read;
    let mut f = fs::File::open(model_path).expect("モデルファイルを開けません");
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 65536];
    loop {
        let n = f.read(&mut buf).expect("モデルファイルの読み込みに失敗");
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    format!("{:x}", hasher.finalize())
}

/// JPEG バイト列の xxHash3-128 ハッシュを計算する
fn compute_image_hash(jpeg_bytes: &[u8]) -> u128 {
    XxHash3_128::oneshot(jpeg_bytes)
}

/// キャッシュ DB を開き、テーブルを作成する
async fn open_cache_db(db_path: &std::path::Path) -> SqlitePool {
    let opts = SqliteConnectOptions::new()
        .create_if_missing(true)
        .filename(db_path);
    let pool = SqlitePool::connect_with(opts)
        .await
        .expect("キャッシュ DB を開けません");
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS face_cache (
            model_hash TEXT NOT NULL,
            image_hash TEXT NOT NULL,
            faces      BLOB NOT NULL,
            PRIMARY KEY (model_hash, image_hash)
        );",
    )
    .execute(&pool)
    .await
    .expect("キャッシュテーブルの作成に失敗");
    pool
}

/// キャッシュから顔検出結果を読み込む。キャッシュミスなら None。
async fn load_cached_faces(
    pool: &SqlitePool,
    model_hash: &str,
    image_hash: u128,
) -> Option<Vec<FoundFace>> {
    let hash_str = format!("{:032x}", image_hash);
    let row: Option<(Vec<u8>,)> = sqlx::query_as(
        "SELECT faces FROM face_cache WHERE model_hash = ? AND image_hash = ?;",
    )
    .bind(model_hash)
    .bind(&hash_str)
    .fetch_optional(pool)
    .await
    .ok()
    .flatten();

    row.map(|(blob,)| {
        let (faces, _): (Vec<FoundFace>, _) =
            bincode::serde::decode_from_slice(&blob, bincode::config::standard())
                .expect("キャッシュのデシリアライズに失敗");
        faces
    })
}

/// 顔検出結果をキャッシュに保存する
async fn save_cached_faces(
    pool: &SqlitePool,
    model_hash: &str,
    image_hash: u128,
    faces: &[FoundFace],
) {
    let hash_str = format!("{:032x}", image_hash);
    let blob = bincode::serde::encode_to_vec(faces, bincode::config::standard())
        .expect("キャッシュのシリアライズに失敗");
    let _ = sqlx::query(
        "INSERT OR REPLACE INTO face_cache (model_hash, image_hash, faces) VALUES (?, ?, ?);",
    )
    .bind(model_hash)
    .bind(&hash_str)
    .bind(&blob)
    .execute(pool)
    .await;
}

// ============================================================================
// 推論スレッド
// ============================================================================

/// TensorRT セッションを構築する
fn build_trt_session() -> Session {
    let s = INFERENCE_SIZE;
    let b = BATCH_SIZE;
    let shape_min = format!("input:1x3x{s}x{s}");
    let shape_opt = format!("input:{b}x3x{s}x{s}");
    let shape_max = format!("input:{b}x3x{s}x{s}");

    Session::builder()
        .unwrap()
        .with_execution_providers([TensorRTExecutionProvider::default()
            .with_max_workspace_size(2 * 1024 * 1024 * 1024) // 2 GB
            .with_fp16(true)
            .with_engine_cache(true)
            .with_engine_cache_path("trt_cache")
            .with_timing_cache(true)
            .with_timing_cache_path("trt_cache")
            .with_build_heuristics(true)
            .with_context_memory_sharing(true)
            .with_cuda_graph(true)
            .with_profile_min_shapes(&shape_min)
            .with_profile_opt_shapes(&shape_opt)
            .with_profile_max_shapes(&shape_max)
            .build()
            .error_on_failure()])
        .unwrap()
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .unwrap()
        .commit_from_file(MODEL_PATH.as_path())
        .unwrap()
}

/// テンソルを推論し、(confidence, bbox, landmark) を返す
fn run_model(
    model: &mut Session,
    tensor: Tensor<f32>,
) -> (
    ndarray::Array<f32, ndarray::IxDyn>,
    ndarray::Array<f32, ndarray::IxDyn>,
    ndarray::Array<f32, ndarray::IxDyn>,
) {
    let extract = |v: &Value| v.try_extract_array::<f32>().unwrap().to_owned();
    let out = model
        .run(inputs! {"input" => tensor})
        .unwrap();
    (
        extract(out.get("confidence").unwrap()),
        extract(out.get("bbox").unwrap()),
        extract(out.get("landmark").unwrap()),
    )
}

/// キャッシュミス画像のみ抽出したサブテンソルを構築する
fn build_miss_tensor(full_tensor: &Tensor<f32>, miss_indices: &[usize]) -> Tensor<f32> {
    let orig_array = full_tensor.try_extract_array::<f32>().unwrap().to_owned();
    let miss_arrays: Vec<_> = miss_indices
        .iter()
        .map(|&i| {
            orig_array
                .index_axis(ndarray::Axis(0), i)
                .insert_axis(ndarray::Axis(0))
                .into_owned()
        })
        .collect();
    let views: Vec<_> = miss_arrays.iter().map(|a| a.view()).collect();
    let concatenated = ndarray::concatenate(ndarray::Axis(0), &views).unwrap();
    Tensor::from_array(
        concatenated
            .into_dimensionality::<ndarray::Ix4>()
            .unwrap(),
    )
    .unwrap()
}

/// 推論スレッドのメインループ。
/// キャッシュヒットはスキップし、ミス分のみ推論→post_process→キャッシュ保存を行い、
/// 統合結果を crop_and_export スレッドに送信する。
async fn inference_loop(
    receiver: Receiver<InferenceBatch>,
    sender: SyncSender<FaceDetectionBatch>,
    model_hash: String,
    cache_db_path: PathBuf,
) {
    ameba_blog_downloader::init_ort();

    let mut model = build_trt_session();

    // post_process 用の検出器（Session はメタデータ参照のみ）
    let detector = RetinaFaceFaceDetector {
        session: Session::builder()
            .unwrap()
            .commit_from_file(MODEL_PATH.as_path())
            .unwrap(),
        model: ModelKind::ResNet,
    };

    let pool = open_cache_db(&cache_db_path).await;

    while let Ok((tensor, paths, image_hashes)) = receiver.recv() {
        // 空パスは終了シグナル
        if paths.is_empty() {
            break;
        }

        let full_shape: Vec<usize> = tensor.shape().iter().map(|&v| v as usize).collect();

        // 各画像のキャッシュを検索
        let mut cached: Vec<Option<Vec<FoundFace>>> = Vec::with_capacity(image_hashes.len());
        for &h in &image_hashes {
            cached.push(load_cached_faces(&pool, &model_hash, h).await);
        }

        let miss_indices: Vec<usize> = cached
            .iter()
            .enumerate()
            .filter_map(|(i, c)| c.is_none().then_some(i))
            .collect();

        // キャッシュミス分を推論＋post_process
        let inferred_faces: Vec<Vec<FoundFace>> = if miss_indices.is_empty() {
            debug!("バッチ全キャッシュヒット ({} 枚)、推論スキップ", paths.len());
            Vec::new()
        } else {
            let (conf, loc, lm) = if miss_indices.len() == paths.len() {
                debug!("バッチ全ミス ({} 枚)、フル推論実行", paths.len());
                run_model(&mut model, tensor)
            } else {
                debug!(
                    "バッチ部分キャッシュ (ヒット {}, ミス {})、部分推論実行",
                    paths.len() - miss_indices.len(),
                    miss_indices.len()
                );
                let miss_tensor = build_miss_tensor(&tensor, &miss_indices);
                run_model(&mut model, miss_tensor)
            };

            let miss_shape = {
                let mut s = full_shape.clone();
                s[0] = miss_indices.len();
                s
            };
            let miss_faces = detector.post_process(conf, loc, lm, miss_shape).unwrap();

            // キャッシュに保存
            for (order, &orig_idx) in miss_indices.iter().enumerate() {
                save_cached_faces(
                    &pool,
                    &model_hash,
                    image_hashes[orig_idx],
                    &miss_faces[order],
                )
                .await;
            }

            miss_faces
        };

        // キャッシュヒットと推論結果を元の順序で統合
        let mut faces_vec: Vec<Vec<FoundFace>> = Vec::with_capacity(paths.len());
        let mut miss_cursor = 0usize;
        for entry in &cached {
            match entry {
                Some(faces) => faces_vec.push(faces.clone()),
                None => {
                    faces_vec.push(inferred_faces[miss_cursor].clone());
                    miss_cursor += 1;
                }
            }
        }

        if sender.send((faces_vec, paths)).is_err() {
            break;
        }
    }
}

// ============================================================================
// 顔クロップ＆エクスポート
// ============================================================================

/// 5 点ランドマークから顔の傾き角を計算する
fn calc_tilt(landmarks: [[f32; 2]; 5]) -> f32 {
    let eye_center = [
        (landmarks[0][0] + landmarks[1][0]) / 2.0,
        (landmarks[0][1] + landmarks[1][1]) / 2.0,
    ];
    let mouth_center = [
        (landmarks[3][0] + landmarks[4][0]) / 2.0,
        (landmarks[3][1] + landmarks[4][1]) / 2.0,
    ];
    f32::atan2(
        eye_center[1] - mouth_center[1],
        eye_center[0] - mouth_center[0],
    )
}

/// デバッグ用: 検出した顔の回転矩形を画像上に描画する
#[allow(dead_code)]
fn draw_rect(mut image: RgbImage, scale: f32, faces: &[FoundFace]) -> RgbImage {
    let inv = 1.0 / scale;
    for face in faces {
        let angle = calc_tilt(face.landmarks) + f32::PI() / 2.0;
        let cx = (face.bbox[0] + face.bbox[2]) * inv / 2.0;
        let cy = (face.bbox[1] + face.bbox[3]) * inv / 2.0;
        let rot = arr2(&[
            [angle.cos(), -angle.sin()],
            [angle.sin(), angle.cos()],
        ]);
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

/// 検出された顔を回転補正してクロップする（100px 未満の顔はスキップ）
fn crop_faces(image: &RgbImage, scale: f32, faces: &[FoundFace]) -> Vec<RgbImage> {
    let inv = 1.0 / scale;
    faces
        .iter()
        .filter(|f| f32::max(f.bbox[2] - f.bbox[0], f.bbox[3] - f.bbox[1]) * inv >= 100.0)
        .map(|face| {
            let angle = calc_tilt(face.landmarks) + f32::PI() / 2.0;
            let cx = (face.bbox[0] + face.bbox[2]) * inv / 2.0;
            let cy = (face.bbox[1] + face.bbox[3]) * inv / 2.0;
            let rotated =
                rotate(image, (cx, cy), -angle, Interpolation::Bilinear, Rgb([0, 0, 0]));
            let size = (f32::max(
                (face.bbox[2] - face.bbox[0]) * inv,
                (face.bbox[3] - face.bbox[1]) * inv,
            ) * 1.2) as u32;
            let half = size as f32 / 2.0;
            crop_imm(
                &rotated,
                (cx - half) as u32,
                (cy - half) as u32,
                size,
                size,
            )
            .to_image()
        })
        .collect()
}

/// クロップ＆エクスポートスレッド。
/// 推論スレッドから受け取った検出結果を元画像と突き合わせ、顔を切り出して保存する。
fn crop_and_export(
    face_rx: Receiver<FaceDetectionBatch>,
    original_rx: Receiver<OriginalImage>,
    file_count: usize,
) {
    let mut originals: HashMap<PathBuf, (RgbImage, f64)> = HashMap::new();
    let mut bar = tqdm!(total = file_count, disable = false);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(16)
        .build()
        .unwrap();
    let export_base = data_dir().join("face_cropped");

    for (faces_vec, paths) in face_rx.iter() {
        if paths.is_empty() {
            return;
        }

        // 元画像チャネルから利用可能な画像を取得
        while let Ok((path, image, scale)) = original_rx.try_recv() {
            originals.insert(path, (image, scale));
        }

        // 出力ディレクトリを確保
        for dir in paths
            .iter()
            .map(|p| p.parent().unwrap().file_name().unwrap())
            .unique()
        {
            let dest = export_base.join(dir);
            if !dest.exists() {
                fs::create_dir_all(&dest).unwrap();
            }
        }

        // 並列でクロップ＆保存
        let work: Vec<_> = faces_vec
            .iter()
            .zip(&paths)
            .map(|(faces, path)| {
                let (image, scale) = &originals[path];
                (image.clone(), *scale as f32, faces.clone(), path.clone())
            })
            .collect();

        pool.install(|| {
            work.into_par_iter()
                .for_each(|(image, scale, faces, path)| {
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

        // 元画像を解放し、プログレスバーを更新
        for path in &paths {
            originals.remove(path);
        }
        for (faces, path) in faces_vec.iter().zip(&paths) {
            debug!("{}", path.display());
            let mut postfix: String = path
                .parent()
                .unwrap()
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .into();
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

// ============================================================================
// デコードヘルパー
// ============================================================================

struct DecodedImage {
    resized: FirImage<'static>,
    original: RgbImage,
    scale: f64,
}

/// JPEG バイト列をデコードし、推論用にリサイズした画像と元画像を返す
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
    decompressor
        .decompress(bin, decoded.as_deref_mut())
        .unwrap();

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
        resizer
            .resize(&fir, &mut dst, &ResizeOptions::default())
            .unwrap();
        dst
    };

    let original = RgbImage::from_raw(fir.width(), fir.height(), fir.into_vec()).unwrap();
    DecodedImage {
        resized,
        original,
        scale: f64::min(1.0, scale),
    }
}

/// リサイズ済み画像群から NCHW テンソル (0-1 正規化, ゼロパディング) を構築する
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

// ============================================================================
// エントリポイント
// ============================================================================

fn main() {
    tracing_subscriber::fmt::init();
    ameba_blog_downloader::init_ort();

    let model_hash = compute_model_hash(MODEL_PATH.as_path().to_str().unwrap());
    debug!("モデルハッシュ (先頭16文字): {}", &model_hash[..16]);

    let cache_db_path = data_dir().join("face_cache.sqlite");

    // 処理対象ファイルを収集
    let mut all_files = Vec::new();
    for entry in data_dir().join("blog_images").read_dir().unwrap() {
        for file in entry.unwrap().path().read_dir().unwrap() {
            all_files.push(file.unwrap().path());
        }
    }

    // チャネル構築
    let (decode_tx, infer_rx) = mpsc::sync_channel::<InferenceBatch>(40);
    let (infer_tx, crop_rx) = mpsc::sync_channel::<FaceDetectionBatch>(40);
    let (orig_tx, orig_rx) = mpsc::channel::<OriginalImage>();

    // 推論スレッド起動
    let mh = model_hash.clone();
    let cd = cache_db_path.clone();
    let inference_handle = thread::spawn(move || {
        tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(inference_loop(infer_rx, infer_tx, mh, cd))
    });

    // クロップ＆エクスポートスレッド起動
    let file_count = all_files.len();
    let crop_handle = thread::spawn(move || crop_and_export(crop_rx, orig_rx, file_count));

    // デコード＆バッチ構築 (rayon 並列)
    all_files
        .chunks(LARGE_BATCH_SIZE)
        .collect::<Vec<_>>()
        .into_par_iter()
        .for_each(|large_chunk| {
            let mut decompressor = Decompressor::new().unwrap();
            let mut resizer = fast_image_resize::Resizer::new();
            unsafe {
                resizer.set_cpu_extensions(fast_image_resize::CpuExtensions::Avx2);
            }

            for files in large_chunk.chunks(BATCH_SIZE) {
                let mut resized_images = Vec::with_capacity(files.len());
                let mut image_hashes = Vec::with_capacity(files.len());

                for file in files {
                    let bin = fs::read(file).unwrap();
                    image_hashes.push(compute_image_hash(&bin));

                    let decoded = decode_and_resize(&bin, &mut decompressor, &mut resizer);
                    orig_tx
                        .send((file.clone(), decoded.original, decoded.scale))
                        .unwrap();
                    resized_images.push(decoded.resized);
                }

                let input = build_padded_tensor(&resized_images);
                decode_tx
                    .send((
                        Tensor::from_array(input).unwrap(),
                        files.to_vec(),
                        image_hashes,
                    ))
                    .unwrap();
            }
            debug!("デコード完了");
        });

    // 終了シグナル送信
    decode_tx
        .send((
            Tensor::from_array(ndarray::array![[[[0.0]]]]).unwrap(),
            vec![],
            vec![],
        ))
        .unwrap();

    inference_handle.join().unwrap();
    crop_handle.join().unwrap();
}
