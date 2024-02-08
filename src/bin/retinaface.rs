use std::fs;
use image::{DynamicImage, ImageResult};
use image::imageops::FilterType;
use ndarray::{Array, ArrayBase, IxDyn, OwnedRepr};
use ort::{Environment, ExecutionProvider, GraphOptimizationLevel, SessionBuilder};
use ort::execution_providers::{CPUExecutionProviderOptions, DirectMLExecutionProviderOptions};
use tract_onnx::prelude::*;

const ONNX_PATH: &str = r#"C:\Users\tomokazu\PycharmProjects\helloproject-ai\retinaface.onnx"#;


fn transform(image: DynamicImage, max_size: usize) -> ArrayBase<OwnedRepr<f32>, IxDyn> {
    let resized_image = image.resize(max_size as u32, max_size as u32, FilterType::Triangle);
    let mut onnx_input = Array::<f32, _>::zeros((1usize, 3usize, max_size, max_size)).into_dyn();
    for (x, y, pixel) in resized_image.into_rgb32f().enumerate_pixels() {
        let [mut r, mut g, mut b] = pixel.0;

        // Normalize
        onnx_input[[0usize, 0, y as usize, x as usize]] = (r - 0.485) / 0.229;
        onnx_input[[0usize, 1, y as usize, x as usize]] = (g - 0.456) / 0.224;
        onnx_input[[0usize, 2, y as usize, x as usize]] = (b - 0.406) / 0.225;
    }
    onnx_input
}

fn main() {
    const MAX_SIZE: usize = 2048;
    const IMAGE_PATH: &str = "manaka_test.jpg";
    let image_bytes = fs::read(IMAGE_PATH).unwrap();


    let environment = Environment::builder()
        .with_name("RetinaFace")
        .with_execution_providers([
            ExecutionProvider::DirectML(DirectMLExecutionProviderOptions::default()),
            ExecutionProvider::CPU(CPUExecutionProviderOptions::default())
        ])
        .build().unwrap().into_arc();

    let session = SessionBuilder::new(&environment).unwrap()
        .with_optimization_level(GraphOptimizationLevel::Level3).unwrap()
        .with_intra_threads(4).unwrap()
        .with_model_from_file(ONNX_PATH).unwrap();

    println!("{:?}", session.inputs);
    println!("{:?}", session.outputs);

    let image;
    match image::load_from_memory(image_bytes.as_slice()) {
        Ok(i) => { image = i }
        Err(err) => {
            eprintln!("Error while loading image: {}", err);
            panic!()
        }
    };
    let transformed_image = transform(image, MAX_SIZE);

    println!("{:?}", transformed_image.shape());
    println!("debug");
}