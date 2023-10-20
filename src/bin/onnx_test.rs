use std::fs;
use image::{GenericImageView};
use ort::{Environment, GraphOptimizationLevel, SessionBuilder, Value};
use ort::ExecutionProvider;
use ndarray::Array;

// 参考: https://github.com/AndreyGermanov/yolov8_onnx_rust/blob/5b28d2550d715f7dbed8ce31b5fdb8e000fa77f6/src/main.rs

fn main() {
    let image_path = r"C:\Users\tomokazu\PycharmProjects\helloproject-ai\稲場愛香=juicejuice-official=12737097989-2.jpg";
    let onnx_path = r"C:\Users\tomokazu\PycharmProjects\helloproject-ai\retinaface.onnx";
    match fs::metadata(onnx_path) {
        Ok(_) => println!("File exist"),
        Err(_) => println!("File not exist")
    }

    tracing_subscriber::fmt::init();
    let environment = Environment::builder()
        .with_name("RetinaFace")
        .with_execution_providers([ExecutionProvider::CPU(Default::default())])
        .build().unwrap()
        .into_arc();

    let session = SessionBuilder::new(&environment).unwrap()
        .with_optimization_level(GraphOptimizationLevel::Level1).unwrap()
        .with_intra_threads(1).unwrap()
        .with_model_from_file(onnx_path).unwrap();

    let image = image::open(image_path).unwrap();
    let (image_width, image_height) = (image.width(), image.height());
    let mut image_arr = Array::zeros((1usize, 3usize, image_height as usize, image_width as usize)).into_dyn();
    for pixel in image.pixels() {
        let x = pixel.0 as usize;
        let y = pixel.1 as usize;
        let [r, g, b, _] = pixel.2.0;
        image_arr[[0, 0, y, x]] = (r as f32) / 255.0;
        image_arr[[0, 1, y, x]] = (g as f32) / 255.0;
        image_arr[[0, 2, y, x]] = (b as f32) / 255.0;
    }
    let image_layout = image_arr.as_standard_layout();
    let onnx_input = vec![Value::from_array(session.allocator(),
                                            &image_layout).unwrap()];
    let model_res = session.run(onnx_input).unwrap();
    println!("{:?}", model_res.get(0));
    println!("{:?}", model_res.get(1));
    println!("{:?}", model_res.get(2));

    println!("{}", image.as_bytes().len());
    return;
}