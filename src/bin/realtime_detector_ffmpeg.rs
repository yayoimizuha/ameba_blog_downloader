use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::process::{exit, Command, Stdio};
use std::sync::mpsc;
use std::sync::mpsc::{Receiver, Sender, SyncSender};
use std::thread;
use itertools::Itertools;
use ndarray::{concatenate, s, stack, Array, ArrayView, Axis, Ix3, IxDyn};
use num_traits::AsPrimitive;
use ort::execution_providers::{CPUExecutionProvider, OpenVINOExecutionProvider};
use ort::inputs;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::{RunOptions, Session};
use rayon::prelude::*;
use fast_image_resize::images::Image as fir_Image;
use fast_image_resize::{CpuExtensions, FilterType, IntoImageView, PixelType, ResizeAlg, ResizeOptions, Resizer};
use fast_image_resize::ResizeAlg::Convolution;
use image::codecs::png::PngEncoder;
use image::{ExtendedColorType, ImageEncoder};
use ort::value::Tensor;

struct YOLO {
    model: Session,
}

impl YOLO {
    fn new() -> YOLO {
        let model = Session::builder().unwrap()
            .with_execution_providers([
                // CPUExecutionProvider::default().build(),
                OpenVINOExecutionProvider::default().with_device_type("GPU").build().error_on_failure(),
            ]).unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3).unwrap()

            .with_intra_threads(1).unwrap();
        YOLO {
            model: model.commit_from_file(r#"C:\Users\tomokazu\RustroverProjects\ameba_blog_downloader\src\bin\yolo11l.onnx"#).unwrap(),
        }
    }
}

const BATCH_SIZE: usize = 16;
const REDUCTION_SCALE: i64 = 2;

fn preprocessor(preprocessor_receiver: Receiver<Option<Vec<u8>>>, preprocessor_sender: SyncSender<Option<Array<f32, Ix3>>>, height: usize, width: usize) {
    let mut cont = true;
    let padded_multiplier = 32;
    while cont {
        // let mut tensors = vec![];
        let buffers = (0..BATCH_SIZE).filter_map(|_| {
            let buffer = if cont {
                preprocessor_receiver.recv().unwrap()
            } else {
                return None
            };
            match buffer {
                None => {
                    cont = false;
                    None
                }
                Some(x) => { Some(x) }
            }
        }).collect::<Vec<_>>();
        let _ = buffers.into_par_iter().map(|buffer| {
            // let buffer = preprocessor_receiver.recv().unwrap();
            let tensor = Array::from_shape_vec(vec![height, width, 4], buffer.clone()).unwrap();
            let tensor = tensor.slice(s![..,..,..3;-1]);
            let shape = tensor.shape();
            let src_image = fir_Image::from_vec_u8(shape[1] as u32, shape[0] as u32, tensor.as_standard_layout().as_slice().unwrap().to_vec(), PixelType::U8x3).unwrap();
            let mut dst_image = fir_Image::new(shape[1].div_ceil(REDUCTION_SCALE as usize) as u32,
                                               shape[0].div_ceil(REDUCTION_SCALE as usize) as u32,
                                               PixelType::U8x3);
            let mut resizer = Resizer::new();
            unsafe { resizer.set_cpu_extensions(CpuExtensions::Avx2); }
            let opt = ResizeOptions::new().resize_alg(Convolution(FilterType::Mitchell));
            resizer.resize(&src_image, &mut dst_image, &opt).unwrap();
            let tensor = Array::from_shape_vec(
                [shape[0].div_ceil(REDUCTION_SCALE as usize),
                    shape[1].div_ceil(REDUCTION_SCALE as usize),
                    3], dst_image.buffer().to_vec()).unwrap().permuted_axes([2, 0, 1]);
            let tensor = tensor.mapv(|v| v as f32 / u8::MAX as f32);
            let shape = tensor.shape();
            // println!("{:?}", shape);

            let mut padded = Array::from_shape_fn([3,
                                                      shape[1].div_ceil(padded_multiplier) * padded_multiplier,
                                                      shape[2].div_ceil(padded_multiplier) * padded_multiplier], |(_, _, _)| { 0f32 });
            padded.slice_mut(s![..,..shape[1],..shape[2]]).assign(&tensor);
            padded
            // println!("{:?}", padded.shape());
        }).map(|tensor| preprocessor_sender.send(Some(tensor)).unwrap()).collect::<Vec<_>>();
    }
    preprocessor_sender.send(None).unwrap()
}

fn infer(infer_receiver: Receiver<Option<Array<f32, Ix3>>>) {
    ort::init().commit().unwrap();
    let mut model = YOLO::new();
    let mut cont = true;
    while cont {
        let mut batch = vec![];
        for _ in 0..BATCH_SIZE {
            match infer_receiver.recv().unwrap() {
                None => {
                    cont = false;
                    break;
                }
                Some(tensor) => {
                    batch.push(tensor);
                }
            }
        }
        let opt = RunOptions::new().unwrap();
        let model_input = inputs! {
                    "images"=>Tensor::from_array(stack(Axis(0),batch.iter().map(|t|t.view()).collect::<Vec<_>>().as_slice()).unwrap()).unwrap()
                };
        let model_resp = model.model.run(model_input).unwrap();
        let feature = model_resp.get("output0").unwrap().try_extract_array::<f32>().unwrap().view().to_owned();
        println!("{:?}", feature.shape());
    }
}
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let ffprobe_output = serde_json::from_str::<serde_json::Value>(
        String::from_utf8(Command::new("ffprobe")
            .args(&[
                args[1].as_str(),
                "-show_streams",
                "-select_streams",
                "v",
                "-count_packets",
                "-print_format",
                "json",
            ])
            .output().unwrap().stdout).unwrap().as_str()).unwrap();
    // println!("{:?}", ffprobe_output);
    let width = ffprobe_output["streams"][0]["width"].as_i64().unwrap() as usize;
    let height = ffprobe_output["streams"][0]["height"].as_i64().unwrap() as usize;
    let total_frames = ffprobe_output["streams"][0]["nb_read_packets"].as_str().unwrap().parse::<i64>().unwrap();
    let frame_rate = ffprobe_output["streams"][0]["r_frame_rate"].as_str().unwrap().to_owned();

    let frame_size = width * height * 4;

    let mut ffmpeg_input_process = Command::new("ffmpeg").args(
        &[
            "-loglevel", "verbose", "-hide_banner",
            "-init_hw_device", "qsv", "-hwaccel", "qsv", "-hwaccel_output_format", "qsv",
            "-i", args[1].as_str(),
            "-vf", "hwupload,vpp_qsv=format=nv12,vpp_qsv=format=bgra,hwdownload",
            "-f", "rawvideo",
            "-c:a", "null",
            "-pix_fmt", "bgra", "-",
        ]
    ).stdout(Stdio::piped()).spawn().unwrap();
    let mut ffmpeg_input_reader = ffmpeg_input_process.stdout.take().unwrap();
    let (main_sender, preprocessor_receiver) = mpsc::sync_channel(BATCH_SIZE * 2);
    let (preprocessor_sender, infer_receiver) = mpsc::sync_channel(BATCH_SIZE * 2);
    let preprocess_proc = thread::spawn(move || {
        preprocessor(preprocessor_receiver, preprocessor_sender, height, width);
    });
    let infer_proc = thread::spawn(move || {
        infer(infer_receiver);
    });
    for _ in 0..total_frames {
        // let mut tensors = vec![];
        let mut buffer = vec![0u8; frame_size];
        ffmpeg_input_reader.read_exact(&mut buffer).unwrap();
        main_sender.send(Some(buffer)).unwrap();
    }
    main_sender.send(None).unwrap();
    preprocess_proc.join().unwrap();
    infer_proc.join().unwrap();
}