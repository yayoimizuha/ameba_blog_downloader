use image::DynamicImage;
use ndarray::{Array, ArrayBase, IxDyn, OwnedRepr};
use ort::execution_providers::{CPUExecutionProviderOptions, CUDAExecutionProviderOptions, TensorRTExecutionProviderOptions};
use ort::ExecutionProvider;
use ort::{Environment, GraphOptimizationLevel, SessionBuilder, Value};
use std::fs;
use image::imageops::FilterType;


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
        .with_model_from_file(
            onnx_path
        ).unwrap();

    let image = image::open(image_path).unwrap();

    let (image_arr, scale) = prepare_image(image.clone(), 640).clone();
    // println!("{}", image_arr);

    let image_layout = image_arr.as_standard_layout();

    let onnx_input = vec![Value::from_array(session.allocator(), &image_layout).unwrap()];

    let model_res = session.run(onnx_input).unwrap();

    let [loc, conf, land] = match &model_res[..] {
        [loc, conf, land, ..] => [loc, conf, land].map(|x| {
            x.try_extract::<f32>().unwrap().view().to_owned()
        }),
        &_ => unreachable!(),
    };
    println!("{:?}", [loc, conf, land]);


    return ();
}
