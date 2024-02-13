use std::fs;
use image::DynamicImage;
use image::imageops::FilterType;
use ndarray::{Array, ArrayBase, Axis, Ix, Ix4, IxDyn, OwnedRepr, s};
use ort::{Environment, ExecutionProvider, GraphOptimizationLevel, NdArrayExtensions, SessionBuilder, Value};
use ort::execution_providers::{CPUExecutionProviderOptions, DirectMLExecutionProviderOptions};
use tract_onnx::prelude::*;

const ONNX_PATH: &str = r#"C:\Users\tomokazu\build\retinaface\retinaface.onnx"#;


fn transform(image: DynamicImage, max_size: usize) -> ArrayBase<OwnedRepr<f32>, IxDyn> {
    let resized_image = image.resize(max_size as u32, max_size as u32, FilterType::Triangle);
    resized_image.save("resized.jpg").unwrap();
    let mut onnx_input = Array::<f32, _>::zeros((1usize, 3usize, max_size, max_size)).into_dyn();
    onnx_input.fill(1.0);
    let resized_width = resized_image.width() as i32;
    let resized_height = resized_image.height() as i32;

    let (x_pad, y_pad) = match resized_width - resized_height {
        x if x == 0 => {
            (0, 0)
        }
        x if x < 0 => {
            ((resized_height - resized_width) / 2, 0)
        }
        x if x > 0 => {
            (0, (resized_width - resized_height) / 2)
        }
        _ => unreachable!()
    };

    for (x, y, pixel) in resized_image.into_rgb32f().enumerate_pixels() {
        let [r, g, b] = pixel.0; // Normalize
        onnx_input[[0usize, 0, y as usize, x as usize]] = (b - 0.485) / 0.229;
        onnx_input[[0usize, 1, y as usize, x as usize]] = (g - 0.456) / 0.224;
        onnx_input[[0usize, 2, y as usize, x as usize]] = (r - 0.406) / 0.225;
    }
    onnx_input
}

fn main() {
    const MAX_SIZE: usize = 1024;
    const IMAGE_PATH: &str = r#"C:\Users\tomokazu\WebstormProjects\hp-face-recognizer-wasm\src\assets\image-sample\entrepreneur-593358_1280.jpg"#;
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

    // println!("{:?}", transformed_image.shape());

    // println!("{}", transformed_image.slice(s![0,..,300,300..304]));
    println!("{}", transformed_image.slice(s![0,..,0,0]));

    let image_layout = transformed_image.as_standard_layout();
    let onnx_input = vec![Value::from_array(session.allocator(), &image_layout).unwrap()];
    let model_res = session.run(onnx_input).unwrap();

    let [loc, conf, land] = match &model_res[..] {
        [loc, conf, land, ..] => [loc, conf, land].map(|x| {
            let arr = x.try_extract::<f32>().unwrap().view().to_owned();
            println!("{:?}", arr.shape());
            arr
        }),
        &_ => unreachable!(),
    };
    println!("{}", loc);
    println!("{}", conf);
    println!("{}", land);
    // let res = conf.softmax(Axis(0));
    // println!("{:?}", res);
    println!("debug");
}