use image::{DynamicImage, Rgb};
use ndarray::{Array, ArrayBase, Ix1, Ix2, IxDyn, OwnedRepr, s};
use ort::execution_providers::{CPUExecutionProviderOptions, CUDAExecutionProviderOptions, TensorRTExecutionProviderOptions};
use ort::ExecutionProvider;
use ort::{Environment, GraphOptimizationLevel, SessionBuilder, Value};
use std::{fmt, fs};
use std::fmt::Formatter;
use image::imageops::FilterType;
use imageproc::geometric_transformations::{Interpolation, rotate};
use imageproc::drawing::draw_hollow_rect_mut;
use imageproc::rect::Rect;
use include_bytes_zstd;

#[derive(Debug)]
struct Face {
    landmark: Array<f32, Ix2>,
    bbox: Array<f32, Ix1>,
    confidence: f32,
}

impl fmt::Display for Face {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}\n", self.landmark).unwrap();
        write!(f, "{}\n", self.bbox).unwrap();
        write!(f, "{}", self.confidence)
    }
}
// 参考: https://github.com/AndreyGermanov/yolov8_onnx_rust/blob/5b28d2550d715f7dbed8ce31b5fdb8e000fa77f6/src/main.rs

fn prepare_image(input: DynamicImage, max_size: u32) -> (ArrayBase<OwnedRepr<f32>, IxDyn>, f32) {
    let resized = input.resize(max_size, max_size, FilterType::Nearest);
    let scale = input.height() as f32 / resized.height() as f32;
    let mut onnx_input =
        Array::<f32, _>::zeros((1usize, 3usize, max_size as usize, max_size as usize)).into_dyn();
    for (x, y, pixel) in resized.into_rgb32f().enumerate_pixels() {
        let [mut r, mut g, mut b] = pixel.0;
        b -= 0.485;
        g -= 0.456;
        r -= 0.406;
        b /= 0.229;
        g /= 0.224;
        r /= 0.225;
        onnx_input[[0usize, 0, y as usize, x as usize]] = b;
        onnx_input[[0usize, 1, y as usize, x as usize]] = g;
        onnx_input[[0usize, 2, y as usize, x as usize]] = r;
    }
    (onnx_input, scale)
}

fn truncate(landmark: Array<f32, Ix2>) -> (f32, f32, f32) {
    let [left_eye, right_eye, _nose, left_mouth, right_mouth] = {
        let mut arr = [[0.0f32; 2]; 5];
        let land_vec = landmark.into_raw_vec().clone();
        for i in 0..5 {
            arr[i] = [land_vec[i * 2], land_vec[i * 2 + 1]];
        }
        arr
    };

    let center_x = [left_eye[0], right_eye[0], left_mouth[0], right_mouth[0]].iter().sum::<f32>() / 4.0;
    let center_y = [left_eye[1], right_eye[1], left_mouth[1], right_mouth[1]].iter().sum::<f32>() / 4.0;
    let eye_center = ((right_eye[0] + left_eye[0]) / 2.0,
                      (right_eye[1] + left_eye[1]) / 2.0);
    let mouth_center = ((right_mouth[0] + left_mouth[0]) / 2.0,
                        (right_mouth[1] + left_mouth[1]) / 2.0);
    return (center_x, center_y, (eye_center.0 - mouth_center.0).atan2(mouth_center.1 - eye_center.1));
}

fn main() -> () {
    let image_path = r"manaka_test.jpg";
    let onnx_path = r"C:\Users\tomokazu\build\retinaface\retinaface_1280.onnx";
    match fs::metadata(onnx_path) {
        Ok(_) => println!("RetinaFace File exist"),
        Err(_) => eprintln!("RetinaFace File not exist"),
    }
    match fs::metadata(image_path) {
        Ok(_) => println!("image File exist"),
        Err(_) => eprintln!("image File not exist"),
    }

    tracing_subscriber::fmt::init();
    let environment = Environment::builder()
        .with_name("RetinaFace")
        .with_execution_providers([
            ExecutionProvider::TensorRT(TensorRTExecutionProviderOptions::default()),
            ExecutionProvider::CUDA(CUDAExecutionProviderOptions::default()),
            ExecutionProvider::CPU(CPUExecutionProviderOptions::default()),
        ]).build().unwrap().into_arc();

    let session = SessionBuilder::new(&environment).unwrap()
        .with_optimization_level(GraphOptimizationLevel::Level1).unwrap()
        .with_intra_threads(1).unwrap()
        .with_model_from_memory(include_bytes!("C:\\Users\\tomokazu\\build\\retinaface\\retinaface_1280.onnx"))
        // .with_model_from_file(onnx_path)
        .unwrap();

    let image = image::open(image_path).unwrap();

    let (image_arr, scale) = prepare_image(image.clone(), 640).clone();

    let image_layout = image_arr.as_standard_layout();

    let onnx_input = vec![Value::from_array(session.allocator(), &image_layout).unwrap()];

    let model_res = session.run(onnx_input).unwrap();

    let [loc, conf, land] = match &model_res[..] {
        [loc, conf, land, ..] => [loc, conf, land].map(|x| {
            x.try_extract::<f32>().unwrap().view().to_owned()
        }),
        &_ => unreachable!(),
    };

    let mut face_list = Vec::<Face>::new();
    for i in 0..loc.clone().shape()[0] {
        face_list.push(Face {
            landmark: Array::from_shape_vec((5, 2), land.slice(s![i,..]).iter().map(|&x| { x * scale }).collect()).unwrap(),
            confidence: conf[i],
            bbox: loc.slice(s![i,..]).iter().map(|&x| { x * scale }).collect::<Array<f32, Ix1>>(),
        });
    }
    println!("{:?}", face_list);
    let mut draw_canvas = image.to_rgb8().clone();
    let mut faces: Vec<_> = vec![];
    for face in &face_list {
        let face_pos = truncate(face.landmark.clone());
        println!("{:?}", face_pos);
        faces.push(DynamicImage::from(rotate(&draw_canvas, (face_pos.0, face_pos.1), -face_pos.2, Interpolation::Bilinear,
                                             Rgb([0, 0, 0]))).crop(face.bbox[0] as u32, face.bbox[1] as u32, face.bbox[2] as u32 - face.bbox[0] as u32, face.bbox[3] as u32 - face.bbox[1] as u32));
        draw_canvas = rotate(&draw_canvas, (face_pos.0, face_pos.1), -face_pos.2, Interpolation::Bilinear, Rgb([0, 0, 0]));
        draw_hollow_rect_mut(&mut draw_canvas, Rect::at(face.bbox[0] as i32, face.bbox[1] as i32).
            of_size(face.bbox[2] as u32 - face.bbox[0] as u32,
                    face.bbox[3] as u32 - face.bbox[1] as u32),
                             Rgb([0, 255, 244]));
        draw_canvas = rotate(&draw_canvas, (face_pos.0, face_pos.1), face_pos.2, Interpolation::Bilinear, Rgb([0, 0, 0]));
    }
    draw_canvas.save("test_rect.jpg").unwrap();
    // println!("{:?}", faces);
    return ();
}
