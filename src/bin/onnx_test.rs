use std::cmp::max;
use image::{DynamicImage, GenericImage, GenericImageView, ImageBuffer, Rgb};
use ndarray::Array;
use ort::execution_providers::{CPUExecutionProviderOptions, CUDAExecutionProviderOptions, TensorRTExecutionProviderOptions};
use ort::ExecutionProvider;
use ort::{Environment, GraphOptimizationLevel, SessionBuilder, Value};
use std::fs;
use image::DynamicImage::ImageRgb8;
use image::imageops::FilterType;
use imageproc::drawing::Canvas;


// 参考: https://github.com/AndreyGermanov/yolov8_onnx_rust/blob/5b28d2550d715f7dbed8ce31b5fdb8e000fa77f6/src/main.rs

fn prepare_image(input: DynamicImage, max_size: u32) -> ImageBuffer<Rgb<f32>, Vec<f32>> {
    let mut resized = input.clone();
    let (input_width, input_height) = (input.width(), input.height());
    if input_width > max_size || input_height > max_size {
        if input_width > input_height {
            resized = input.resize(max_size, max_size, FilterType::Nearest);
        }
        if input_height >= input_width {
            resized = input.resize(max_size, max_size, FilterType::Nearest);
        }
    }
    let mut canvas =
        ImageBuffer::from_pixel(max_size, max_size, Rgb::<f32>([0f32; 3]));
    let (resized_width, resized_height) = (resized.width(), resized.height());
    for (x, y, pixel) in resized.into_rgb32f().enumerate_pixels() {
        let [mut r, mut g, mut b] = pixel.0;
        if x == 0 && y == 0 { println!("{:?}", pixel) }
        r -= 0.485;
        g -= 0.456;
        b -= 0.406;
        r /= 0.229;
        g /= 0.224;
        b /= 0.225;
        canvas.put_pixel((max_size - resized_width) / 2 + x, y,
                         Rgb([r, g, b]));
    }
    canvas
}

fn main() -> () {
    let image_path = r"manaka_test.jpg";
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

    let image = image::open(image_path).unwrap();
    // println!("{:?}", Canvas::get_pixel(&image, 0, 0));
    let image = prepare_image(image.clone(), 640).clone();
    DynamicImage::from(image.clone()).to_rgb8().save("test.jpg").unwrap();

    let (image_width, image_height) = (image.width(), image.height());

    let mut image_arr =
        Array::<f32, _>::zeros((1usize, 3usize, image_height as usize, image_width as usize))
            .into_dyn();
    for pixel in image.enumerate_pixels() {
        let x = pixel.0 as usize;
        let y = pixel.1 as usize;
        let [r, g, b] = pixel.2.0;
        image_arr[[0, 0, y, x]] = (r as f32) / 1.0;
        image_arr[[0, 1, y, x]] = (g as f32) / 1.0;
        image_arr[[0, 2, y, x]] = (b as f32) / 1.0;
    }
    let image_layout = image_arr.as_standard_layout();

    let onnx_input = vec![Value::from_array(session.allocator(), &image_layout).unwrap()];
    println!("{:?}", onnx_input);
    // println!("{}", onnx_input.get(0).unwrap().try_extract::<f32>().unwrap().view().clone().into_owned());
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
