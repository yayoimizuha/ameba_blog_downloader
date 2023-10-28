use image::{DynamicImage, Rgb};
use ndarray::{Array, ArrayBase, Axis, Ix1, Ix2, IxDyn, OwnedRepr, s};
use ort::execution_providers::{CPUExecutionProviderOptions, CUDAExecutionProviderOptions, TensorRTExecutionProviderOptions};
use ort::{ExecutionProvider, NdArrayExtensions};
use ort::{Environment, GraphOptimizationLevel, SessionBuilder, Value};
use std::{fmt, fs};
use std::fmt::Formatter;
use image::imageops::FilterType;
use imageproc::geometric_transformations::{Interpolation, rotate};
use imageproc::drawing::draw_hollow_rect_mut;
use imageproc::rect::Rect;
// use include_bytes_zstd;

#[derive(Debug)]
struct Face {
    landmark: Array<f32, Ix2>,
    bbox: Array<f32, Ix1>,
    confidence: f32,
}

const NAME_LIST: [&str; 118] = ["一岡伶奈", "上國料萌衣", "下井谷幸穂", "中山夏月姫", "中島早貴", "中西香菜", "井上春華", "井上玲音", "伊勢鈴蘭",
    "佐々木莉佳子", "佐藤優樹", "入江里咲", "八木栞", "前田こころ", "加賀楓", "勝田里奈", "北原もも", "北川莉央", "和田彩花",
    "和田桜子", "嗣永桃子", "夏焼雅", "太田遥香", "室田瑞希", "宮崎由加", "宮本佳林", "小川麗奈", "小林萌花", "小片リサ",
    "小田さくら", "小野瑞歩", "小野田紗栞", "小関舞", "尾形春水", "山岸理子", "山木梨沙", "山﨑夢羽", "山﨑愛生",
    "岡井千聖", "岡村ほまれ", "岡村美波", "岸本ゆめの", "島倉りか", "川名凜", "川嶋美楓", "川村文乃", "工藤由愛",
    "工藤遥", "平井美葉", "平山遊季", "広本瑠璃", "広瀬彩海", "弓桁朱琴", "後藤花", "徳永千奈美", "斉藤円香", "新沼希空",
    "有澤一華", "松本わかな", "松永里愛", "梁川奈々美", "森戸知沙希", "植村あかり", "横山玲奈", "橋迫鈴", "櫻井梨央", "段原瑠々",
    "江口紗耶", "江端妃咲", "河西結心", "浅倉樹々", "浜浦彩乃", "清水佐紀", "清野桃々姫", "為永幸音", "熊井友理奈", "牧野真莉愛",
    "生田衣梨奈", "田中れいな", "田代すみれ", "田口夏実", "田村芽実", "相川茉穂", "矢島舞美", "石山咲良", "石栗奏美", "石田亜佑美",
    "福田真琳", "秋山眞緒", "稲場愛香", "窪田七海", "竹内朱莉", "笠原桃奈", "筒井澪心", "米村姫良々", "羽賀朱音", "船木結",
    "菅谷梨沙子", "萩原舞", "藤井梨央", "西田汐里", "西﨑美空", "譜久村聖", "谷本安美", "豫風瑠乃", "道重さゆみ", "遠藤彩加里",
    "里吉うたの", "野中美希", "野村みな美", "金澤朋子", "鈴木愛理", "鈴木香音", "鞘師里保", "須藤茉麻", "飯窪春菜", "高木紗友希", "高瀬くるみ"];

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
        .with_model_from_memory(include_bytes!("retinaface_sim.onnx"))
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
        if f32::max(face_pos.0, face_pos.1) > 80f32 {
            faces.push(DynamicImage::from(rotate(&draw_canvas, (f32::max(face_pos.0, face_pos.1), f32::max(face_pos.0, face_pos.1)), -face_pos.2, Interpolation::Bilinear,
                                                 Rgb([0, 0, 0]))).crop(face.bbox[0] as u32, face.bbox[1] as u32, face.bbox[2] as u32 - face.bbox[0] as u32, face.bbox[3] as u32 - face.bbox[1] as u32));
        }

        draw_canvas = rotate(&draw_canvas, (face_pos.0, face_pos.1), -face_pos.2, Interpolation::Bilinear, Rgb([0, 0, 0]));
        draw_hollow_rect_mut(&mut draw_canvas, Rect::at(face.bbox[0] as i32, face.bbox[1] as i32).
            of_size(face.bbox[2] as u32 - face.bbox[0] as u32,
                    face.bbox[3] as u32 - face.bbox[1] as u32),
                             Rgb([0, 255, 244]));
        draw_canvas = rotate(&draw_canvas, (face_pos.0, face_pos.1), face_pos.2, Interpolation::Bilinear, Rgb([0, 0, 0]));
    }
    draw_canvas.save("test_rect.jpg").unwrap();
    let mut face_arr = Array::<f32, _>::zeros((faces.len(), 3usize, 224usize, 224usize)).into_dyn();
    for i in 0..faces.len() {
        if faces[i].width() > 80 {
            for (x, y, pixel) in faces[i].resize(224, 224, FilterType::Gaussian).into_rgb32f().enumerate_pixels() {
                face_arr[[i, 0, y as usize, x as usize]] = pixel.0[0];
                face_arr[[i, 1, y as usize, x as usize]] = pixel.0[1];
                face_arr[[i, 2, y as usize, x as usize]] = pixel.0[2];
            }
        }
    }
    let recognition_environment = Environment::builder()
        .with_name("RecognitionResNet")
        .with_execution_providers([
            ExecutionProvider::TensorRT(TensorRTExecutionProviderOptions::default()),
            ExecutionProvider::CUDA(CUDAExecutionProviderOptions::default()),
            ExecutionProvider::CPU(CPUExecutionProviderOptions::default()),
        ]).build().unwrap().into_arc();

    let recognition_session = SessionBuilder::new(&recognition_environment).unwrap()
        .with_optimization_level(GraphOptimizationLevel::Level1).unwrap()
        .with_intra_threads(1).unwrap()
        .with_model_from_memory(include_bytes!(r"face_recognition_sim.onnx"))
        // .with_model_from_file(onnx_path)
        .unwrap();

    let recognition_layout = face_arr.as_standard_layout();
    let recognition_input = vec![(Value::from_array(recognition_session.allocator(), &recognition_layout)).unwrap()];
    let recognition_res = recognition_session.run(recognition_input).unwrap();
    for i in recognition_res[0].try_extract::<f32>().unwrap().view().to_owned().axis_iter(Axis(0)) {
        let mut row = i.softmax(Axis(0)).into_iter().enumerate().collect::<Vec<_>>();
        row.sort_by(|a, b| { (-a.1).partial_cmp(&-b.1).unwrap() });
        println!("\n");
        &row[..5].iter().for_each(|x| {
            println!("{}: {}%", NAME_LIST[x.0], &x.1 * 100f32);
        });
    }

    // println!("{:?}", faces);
    return ();
}
