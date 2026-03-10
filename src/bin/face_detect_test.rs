/// 顔検出テスト: ResNet / MobileNet 両モデルで画像上の顔にバウンディングボックスを描画する
///
/// 使い方:
///   cargo run --release --bin face_detect_test -- <input_image_path>
///
/// 出力:
///   <input_basename>_resnet.jpg  -- ResNet モデルの検出結果
///   <input_basename>_mobilenet.jpg -- MobileNet モデルの検出結果
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::time::Instant;

use image::{DynamicImage, ImageFormat, Rgb, RgbImage};
use image::imageops::FilterType;
use imageproc::drawing::draw_hollow_rect_mut;
use imageproc::rect::Rect;
use ort::execution_providers::CPUExecutionProvider;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;

use ameba_blog_downloader::project_dir;
use ameba_blog_downloader::retinaface::retinaface_common::{ModelKind, RetinaFaceFaceDetector};
use ameba_blog_downloader::retinaface::found_face::FoundFace;

/// 推論用の最大サイズ (長辺がこのサイズに収まるようリサイズ)
const INFERENCE_SIZE: u32 = 640;

fn draw_faces(img: &RgbImage, faces: &[FoundFace], color: Rgb<u8>) -> RgbImage {
    let mut canvas = img.clone();
    for face in faces {
        let x1 = face.bbox[0].max(0.0) as i32;
        let y1 = face.bbox[1].max(0.0) as i32;
        let x2 = (face.bbox[2] as i32).min(canvas.width() as i32 - 1);
        let y2 = (face.bbox[3] as i32).min(canvas.height() as i32 - 1);
        let w = (x2 - x1).max(0);
        let h = (y2 - y1).max(0);
        if w > 0 && h > 0 {
            let rect = Rect::at(x1, y1).of_size(w as u32, h as u32);
            // 太さを出すため3回描く (1px ずつオフセット)
            draw_hollow_rect_mut(&mut canvas, rect, color);
            if w > 2 && h > 2 {
                let rect_inner = Rect::at(x1 + 1, y1 + 1).of_size((w - 2) as u32, (h - 2) as u32);
                draw_hollow_rect_mut(&mut canvas, rect_inner, color);
            }
            if w > 4 && h > 4 {
                let rect_outer = Rect::at((x1 - 1).max(0), (y1 - 1).max(0))
                    .of_size((w + 2).min(canvas.width() as i32) as u32, (h + 2).min(canvas.height() as i32) as u32);
                draw_hollow_rect_mut(&mut canvas, rect_outer, color);
            }
        }
    }
    canvas
}

/// OpenVINO を使わず CPU で Session を構築し、RetinaFaceFaceDetector を作成する
fn build_detector(model_kind: ModelKind, model_path: &Path) -> RetinaFaceFaceDetector {
    let session = Session::builder().unwrap()
        .with_execution_providers([CPUExecutionProvider::default().build()]).unwrap()
        .with_optimization_level(GraphOptimizationLevel::Level1).unwrap()
        .with_intra_threads(4).unwrap()
        .commit_from_file(model_path).unwrap();
    RetinaFaceFaceDetector {
        session,
        model: model_kind,
    }
}

/// 画像を INFERENCE_SIZE に収まるようリサイズし、JPEG バイト列として返す。
/// 元画像が既に収まっている場合はそのままの bytes を返す。
fn resize_image_bytes(original: &DynamicImage) -> Vec<u8> {
    let (w, h) = (original.width(), original.height());
    let max_dim = w.max(h);
    let resized = if max_dim > INFERENCE_SIZE {
        let scale = INFERENCE_SIZE as f64 / max_dim as f64;
        let new_w = (w as f64 * scale).round() as u32;
        let new_h = (h as f64 * scale).round() as u32;
        original.resize_exact(new_w, new_h, FilterType::Triangle)
    } else {
        original.clone()
    };
    let mut buf = Cursor::new(Vec::new());
    resized.write_to(&mut buf, ImageFormat::Jpeg).unwrap();
    buf.into_inner()
}

fn detect_and_draw(
    model_kind: ModelKind,
    model_name: &str,
    model_path: PathBuf,
    original_dynamic: &DynamicImage,
    original_image: &RgbImage,
    output_path: &Path,
) {
    println!("[{model_name}] Loading model from: {}", model_path.display());
    let now = Instant::now();
    let mut detector = build_detector(model_kind, &model_path);
    println!("[{model_name}] Model loaded in {:?}", now.elapsed());

    // 推論用にリサイズした画像バイト列を作成
    let now = Instant::now();
    let resized_bytes = resize_image_bytes(original_dynamic);
    println!("[{model_name}] Resize: {:?}", now.elapsed());

    // image_to_array は Vec<u8> (画像ファイルバイト列) を受け取る
    let now = Instant::now();
    let input_tensor = detector.image_to_array(resized_bytes).unwrap();
    let input_shape = input_tensor.shape().to_vec();
    println!("[{model_name}] Input tensor shape: {:?} (transform: {:?})", input_shape, now.elapsed());

    let now = Instant::now();
    let faces_batch = detector.find_face(input_tensor);
    println!("[{model_name}] Inference + post-process: {:?}", now.elapsed());

    // find_face はバッチ出力 (ここではバッチ=1)
    let faces = &faces_batch[0];
    println!("[{model_name}] Detected {} face(s):", faces.len());
    for (i, face) in faces.iter().enumerate() {
        println!(
            "  Face {}: bbox=[{:.1}, {:.1}, {:.1}, {:.1}], score={:.4}",
            i, face.bbox[0], face.bbox[1], face.bbox[2], face.bbox[3], face.score
        );
    }

    // bbox 座標は推論時の画像サイズ基準なので、元画像サイズにスケールする
    let infer_h = input_shape[2] as f32;
    let infer_w = input_shape[3] as f32;
    let orig_w = original_image.width() as f32;
    let orig_h = original_image.height() as f32;
    let scale_x = orig_w / infer_w;
    let scale_y = orig_h / infer_h;

    let scaled_faces: Vec<FoundFace> = faces
        .iter()
        .map(|f| FoundFace {
            bbox: [
                f.bbox[0] * scale_x,
                f.bbox[1] * scale_y,
                f.bbox[2] * scale_x,
                f.bbox[3] * scale_y,
            ],
            score: f.score,
            landmarks: f.landmarks.map(|[lx, ly]| [lx * scale_x, ly * scale_y]),
        })
        .collect();

    let color = match model_name {
        "ResNet" => Rgb([255, 0, 0]),     // 赤
        _ => Rgb([0, 255, 0]),            // 緑
    };
    let result = draw_faces(original_image, &scaled_faces, color);
    result.save(output_path).unwrap();
    println!("[{model_name}] Result saved to: {}", output_path.display());
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <input_image_path>", args[0]);
        std::process::exit(1);
    }
    let input_path = PathBuf::from(&args[1]);
    if !input_path.exists() {
        eprintln!("Error: File not found: {}", input_path.display());
        std::process::exit(1);
    }

    // GPU provider DLL (CUDA/TensorRT/OpenVINO) がロードされてハングするのを防ぐため、
    // onnxruntime.dll だけを隔離ディレクトリにコピーし ORT_DYLIB_PATH で指定する。
    {
        let release_deps = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("release")
            .join("deps");
        let src_dll = release_deps.join("onnxruntime.dll");
        if src_dll.exists() {
            let isolated_dir = release_deps.join("ort_cpu_only");
            let _ = std::fs::create_dir_all(&isolated_dir);
            let dest_dll = isolated_dir.join("onnxruntime.dll");
            if !dest_dll.exists() || std::fs::metadata(&dest_dll).unwrap().len() != std::fs::metadata(&src_dll).unwrap().len() {
                std::fs::copy(&src_dll, &dest_dll).unwrap();
            }
            std::env::set_var("ORT_DYLIB_PATH", &dest_dll);
        }
    }

    // ORT 初期化
    println!("Initializing ONNX Runtime...");
    ort::init().commit();
    println!("ONNX Runtime initialized.");

    // 画像ファイルを読み込み
    let original_dynamic = image::open(&input_path).unwrap();
    let original_image = original_dynamic.to_rgb8();
    println!(
        "Input image: {} ({}x{})",
        input_path.display(),
        original_image.width(),
        original_image.height()
    );

    let stem = input_path.file_stem().unwrap().to_str().unwrap();
    let parent = input_path.parent().unwrap_or(Path::new("."));

    // モデルパス
    let retinaface_dir = project_dir()
        .join("src")
        .join("retinaface");
    let resnet_model = retinaface_dir.join("resnet_retinaface.onnx");
    let mobilenet_model = retinaface_dir.join("mobilenet_retinaface.onnx");

    // ResNet
    detect_and_draw(
        ModelKind::ResNet,
        "ResNet",
        resnet_model,
        &original_dynamic,
        &original_image,
        &parent.join(format!("{stem}_resnet.jpg")),
    );

    println!();

    // MobileNet
    detect_and_draw(
        ModelKind::MobileNet,
        "MobileNet",
        mobilenet_model,
        &original_dynamic,
        &original_image,
        &parent.join(format!("{stem}_mobilenet.jpg")),
    );

    println!("\nDone!");
}
