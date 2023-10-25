use std::cmp::max;
use image::{DynamicImage, GenericImage, GenericImageView, ImageBuffer, Rgb, Rgb32FImage};
use ndarray::{Array, ArrayBase, IxDyn, OwnedRepr};
use ort::execution_providers::{CPUExecutionProviderOptions, CUDAExecutionProviderOptions, TensorRTExecutionProviderOptions};
use ort::ExecutionProvider;
use ort::{Environment, GraphOptimizationLevel, SessionBuilder, Value};
use std::fs;
use image::DynamicImage::ImageRgb8;
use image::imageops::FilterType;
use imageproc::drawing::Canvas;


// 参考: https://github.com/AndreyGermanov/yolov8_onnx_rust/blob/5b28d2550d715f7dbed8ce31b5fdb8e000fa77f6/src/main.rs

fn prepare_image(input: DynamicImage, max_size: u32) -> ArrayBase<OwnedRepr<f32>, IxDyn> {
    let mut resized = input.clone();
    let (input_width, input_height) = (input.width(), input.height());
    resized = input.resize(max_size, max_size, FilterType::Nearest);
    // if input_width > input_height {}
    // if input_height >= input_width {
    //     resized = input.resize(max_size, max_size, FilterType::Nearest);
    // };

    let mut canvas = Rgb32FImage::new(max_size, max_size);

    let (resized_width, resized_height) = (resized.width(), resized.height());
    let mut onnx_input =
        Array::<f32, _>::zeros((1usize, 3usize, max_size as usize, max_size as usize)).into_dyn();
    for (x, y, pixel) in resized.into_rgb32f().enumerate_pixels() {
        let [mut r, mut g, mut b] = pixel.0;
        if x == 0 && y == 0 { println!("{:?}", pixel) }
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
    onnx_input
}

fn main() -> () {
    let image_path = r"rgb.png";
    let onnx_path = r"C:\Users\tomokazu\PycharmProjects\helloproject-ai\retinaface.onnx";
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
        //.with_execution_providers([ExecutionProvider::CUDA(CUDAExecutionProviderOptions::default())])
        .with_execution_providers([
            ExecutionProvider::TensorRT(TensorRTExecutionProviderOptions::default()),
            ExecutionProvider::CUDA(CUDAExecutionProviderOptions::default()),
            // ExecutionProvider::DirectML(DirectMLExecutionProviderOptions::default()),
            // ExecutionProvider::OpenVINO(OpenVINOExecutionProviderOptions::default()),
            ExecutionProvider::CPU(CPUExecutionProviderOptions::default()),
        ])
        .build()
        .unwrap()
        .into_arc();

    let session = SessionBuilder::new(&environment).unwrap()
        .with_optimization_level(GraphOptimizationLevel::Level1).unwrap()
        .with_intra_threads(1).unwrap()
        .with_model_from_file(onnx_path).unwrap();

    let image: DynamicImage = image::open(image_path).unwrap();
    // println!("{:?}", Canvas::get_pixel(&image, 0, 0));
    let image_arr = prepare_image(image.clone(), 5).clone();
    println!("{}", image_arr);
    // DynamicImage::from(image.clone()).to_rgb8().save("test.jpg").unwrap();

    // let (image_width, image_height) = (image.width(), image.height());

    // let mut image_arr =
    //     Array::<f32, _>::zeros((1usize, 3usize, image_height as usize, image_width as usize))
    //         .into_dyn();
    // for (x, y, pixel) in image.pixels() {
    //     let [r, g, b, _] = pixel.0;
    //     image_arr[[0, 0, y as usize, x as usize]] = (r as f32) / 1.0;
    //     image_arr[[0, 1, y as usize, x as usize]] = (g as f32) / 1.0;
    //     image_arr[[0, 2, y as usize, x as usize]] = (b as f32) / 1.0;
    // }
    // for (x, y, pixel) in image.enumerate_pixels() {
    //     for i in 0..2usize {
    //         image_arr[[0, i, y as usize, x as usize]] = pixel.0[i];
    //     }
    // }
    let image_layout = image_arr.as_standard_layout();

    let onnx_input = vec![Value::from_array(session.allocator(), &image_layout).unwrap()];
    println!("{:?}", onnx_input);
    println!("{}", onnx_input.get(0).unwrap().try_extract::<f32>().unwrap().view().clone().into_owned());
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
