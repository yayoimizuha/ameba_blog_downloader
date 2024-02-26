use std::{f32, fs};
use std::env::args;
use std::ops::{Div, Mul};
use std::time::Instant;
use image::DynamicImage;
use itertools::{enumerate, iproduct};
use ndarray::{arr2, Array, array, Array2, Array4, Axis, concatenate, Ix1, Ix2, s};
use ort::{ArrayExtensions, CPUExecutionProvider, CUDAExecutionProvider, DirectMLExecutionProvider, ExecutionProvider, GraphOptimizationLevel, inputs, Session, TensorRTExecutionProvider, Value};
use powerboxesrs::nms::nms;
use serde::{Deserialize, Serialize};

const ONNX_PATH: &str = r#"C:\Users\tomokazu\RustroverProjects\ameba_blog_downloader\src\bin\resnet_retinaface.onnx"#;


fn transform(image: DynamicImage, _max_size: usize) -> Array4<f32> {
    // let resized_image = if max_size < image.width() as usize || max_size < image.height() as usize {
    //     // image.resize(max_size as u32, max_size as u32, FilterType::Triangle)
    //     image
    // } else {
    //     // image.resize(max_size as u32, max_size as u32, FilterType::Triangle)
    //     image
    // };
    // // resized_image.save("prepare.jpg").unwrap();
    let mut output_image = Array4::zeros((1usize, 3usize, image.height() as usize, image.width() as usize));
    output_image.fill(0.);

    for (x, y, pixel) in image.into_rgb32f().enumerate_pixels() {
        let [r, g, b] = pixel.0; // Normalize
        output_image[[0usize, 0, y as usize, x as usize]] = (r - 0.485) / 0.229;
        output_image[[0usize, 1, y as usize, x as usize]] = (g - 0.456) / 0.224;
        output_image[[0usize, 2, y as usize, x as usize]] = (b - 0.406) / 0.225;
    }
    // for (x, y, pixel) in resized_image.into_rgb8().enumerate_pixels() {
    //     let [r, g, b] = pixel.0; // Normalize
    //     output_image[[0usize, 0, y as usize, x as usize]] = (r as f32 - 0.485 * 255.0) / (0.229 * 255.0);
    //     output_image[[0usize, 1, y as usize, x as usize]] = (g as f32 - 0.456 * 255.0) / (0.224 * 255.0);
    //     output_image[[0usize, 2, y as usize, x as usize]] = (b as f32 - 0.406 * 255.0) / (0.225 * 255.0);
    // }
    // Rgb32FImage::from_fn(max_size as u32, max_size as u32, |x, y| {
    //     let red = output_image[[0usize, 0usize, y as usize, x as usize]];
    //     let green = output_image[[0usize, 1usize, y as usize, x as usize]];
    //     let blue = output_image[[0usize, 2usize, y as usize, x as usize]];
    //     Rgb([red, green, blue])
    // }).save("prepare2.exr").unwrap();
    output_image
}

fn prior_box(min_sizes: Vec<Vec<usize>>, steps: Vec<usize>, clip: bool, image_size: [usize; 2]) -> (Array2<f32>, usize) {
    let feature_maps = steps.iter().map(|&step| {
        [f32::ceil(image_size[0] as f32 / step as f32) as i32,
            f32::ceil(image_size[1] as f32 / step as f32) as i32]
    }).collect::<Vec<_>>();
    // println!("{:?}", feature_maps);
    let mut anchors: Vec<[f32; 4]> = vec![];
    for (k, f) in enumerate(feature_maps) {
        for (i, j) in iproduct!(0..f[0],0..f[1]) {
            let t_min_sizes = &min_sizes[k];
            for &min_size in t_min_sizes {
                let s_kx = min_size as f32 / image_size[0] as f32;
                let s_ky = min_size as f32 / image_size[1] as f32;
                let dense_cx = [j as f32 + 0.5].iter().map(|x| x * steps[k] as f32 / image_size[0] as f32).collect::<Vec<_>>();
                let dense_cy = [i as f32 + 0.5].iter().map(|y| y * steps[k] as f32 / image_size[1] as f32).collect::<Vec<_>>();
                for (cy, cx) in iproduct!(dense_cy,dense_cx) {
                    anchors.push([cx, cy, s_kx, s_ky]);
                }
            }
        }
    }
    let mut output = arr2(&anchors);
    if clip {
        output = output.mapv(|x| f32::min(f32::max(x, 0.0), 1.0));
    }
    (output, anchors.len())
}

fn decode(loc: Array<f32, Ix2>, priors: Array<f32, Ix2>, variances: [f32; 2]) -> Array<f32, Ix2> {
    let mut boxes = concatenate(Axis(1), &*vec![
        (priors.slice(s![..,..2]).to_owned() + loc.slice(s![..,..2]).mul(variances[0]) * priors.slice(s![..,2..])).view(),
        (priors.slice(s![..,2..]).to_owned() * loc.slice(s![..,2..]).mul(variances[1]).to_owned().mapv(f32::exp)).view(),
    ]).unwrap();


    let boxes_sub = boxes.slice(s![..,..2]).to_owned() - boxes.slice(s![..,2..]).div(2.0);
    boxes.slice_mut(s![..,..2]).assign(&boxes_sub);

    let boxes_add = boxes.slice(s![..,2..]).to_owned() + boxes.slice(s![..,..2]);
    boxes.slice_mut(s![..,2..]).assign(&boxes_add);
    boxes
}


fn decode_landmark(pre: Array<f32, Ix2>, priors: Array<f32, Ix2>, variances: [f32; 2]) -> Array<f32, Ix2> {
    return concatenate(Axis(1),
                       &*vec![
                           (priors.slice(s![..,..2]).to_owned() + pre.slice(s![..,..2]).mapv(|x| x * variances[0]) * priors.slice(s![..,2..])).view(),
                           (priors.slice(s![..,..2]).to_owned() + pre.slice(s![..,2..4]).mapv(|x| x * variances[0]) * priors.slice(s![..,2..])).view(),
                           (priors.slice(s![..,..2]).to_owned() + pre.slice(s![..,4..6]).mapv(|x| x * variances[0]) * priors.slice(s![..,2..])).view(),
                           (priors.slice(s![..,..2]).to_owned() + pre.slice(s![..,6..8]).mapv(|x| x * variances[0]) * priors.slice(s![..,2..])).view(),
                           (priors.slice(s![..,..2]).to_owned() + pre.slice(s![..,8..10]).mapv(|x| x * variances[0]) * priors.slice(s![..,2..])).view(),
                       ]).unwrap();
}

#[derive(Serialize, Deserialize, Debug)]
pub struct FoundFace {
    bbox: [f32; 4],
    score: f32,
    landmarks: [[f32; 2]; 5],
}

pub fn infer(image_bytes: Vec<u8>) -> Result<Vec<FoundFace>, String> {
    const MAX_SIZE: usize = 640;

    let execution_providers = [
        TensorRTExecutionProvider::default().build(),
        CUDAExecutionProvider::default().build(),
        DirectMLExecutionProvider::default().build(),
        CPUExecutionProvider::default().build(),
    ];
    for execution_provider in &execution_providers {
        if execution_provider.is_available().unwrap() {
            println!("{}", execution_provider.as_str());
            break;
        }
    }
    let session = Session::builder().unwrap()
        .with_execution_providers(execution_providers).unwrap()
        .with_optimization_level(GraphOptimizationLevel::Level3).unwrap()
        .with_intra_threads(16).unwrap()
        .with_model_from_file(ONNX_PATH).unwrap();


    let confidence_threshold = 0.7;
    let nms_threshold = 0.4;
    let variance = [0.1, 0.2];

    let image;
    match image::load_from_memory(image_bytes.as_slice()) {
        Ok(i) => { image = i }
        Err(err) => {
            eprintln!("Error while loading image: {}", err);
            return Err(err.to_string());
        }
    };
    let raw_image = transform(image, MAX_SIZE);
    println!("{:?}", raw_image.dim());
    let dims = raw_image.shape();
    let mut arr = vec![];
    for i in 0..dims[0] {
        arr.push(vec![]);
        for j in 0..dims[1] {
            arr[i].push(vec![]);
            for k in 0..dims[2] {
                arr[i][j].push(vec![]);
                for l in 0..dims[3] {
                    arr[i][j][k].push(raw_image[[i, j, k, l]]);
                }
            }
        }
    }
    fs::write("serialized_vec.txt", serde_json::to_string(&arr).unwrap()).unwrap();

    let binding = raw_image.to_owned();
    let input_shape = binding.shape();
    let onnx_input = inputs!["input"=>raw_image.view()].unwrap();
    let transformed_size = array![input_shape[2], input_shape[3]].to_owned();

    // println!("{}", raw_image);

    let now = Instant::now();
    let model_res = session.run(onnx_input).unwrap();
    println!("ONNX Inference time: {:?}", now.elapsed());

    let extract = |tensor: &Value| tensor.extract_tensor::<f32>().unwrap().view().to_owned();
    let [ confidence, loc, landmark] = ["confidence", "bbox", "landmark"].map(|label| extract(model_res.get(label).unwrap()));

    let scale_landmarks = concatenate(Axis(0), &*vec![transformed_size.view(); 5]).unwrap().mapv(|x| x as f32);
    let scale_bboxes = concatenate(Axis(0), &*vec![transformed_size.view(); 2]).unwrap().mapv(|x| x as f32);

    let (prior_box, _onnx_output_width) = prior_box(
        vec![vec![16, 32], vec![64, 128], vec![256, 512]],
        [8, 16, 32].into(),
        false,
        [input_shape[2], input_shape[3]],
    );


    let confidence = confidence.softmax(Axis(2));

    let mut boxes = decode(loc.slice(s![0,..,..]).to_owned(), prior_box.clone(), variance);
    boxes = boxes * scale_bboxes;

    let mut scores = confidence.slice(s![0,..,1]).to_owned() as Array<f32, Ix1>;

    let mut landmarks = decode_landmark(landmark.slice(s![0,..,..]).to_owned(), prior_box.clone(), variance);
    landmarks = landmarks * scale_landmarks;


    let valid_index = scores.iter().enumerate().filter(|(_, val)| val > &&confidence_threshold).map(|(order, _)| order).collect::<Vec<_>>();
    boxes = boxes.select(Axis(0), &*valid_index);
    landmarks = landmarks.select(Axis(0), &*valid_index);
    scores = scores.select(Axis(0), &*valid_index);


    let keep = nms(&boxes.to_owned(), &scores.mapv(|x| x as f64).to_owned(), nms_threshold, confidence_threshold as f64);


    let mut faces = vec![];
    for index in keep {
        faces.push(FoundFace {
            bbox: <[f32; 4]>::try_from(boxes.slice(s![index,..]).to_vec()).unwrap(),
            score: *scores.get(index).unwrap(),
            landmarks: <[[f32; 2]; 5]>::try_from(landmarks.slice(s![index,..]).to_vec().chunks_exact(2).map(|x| { <[f32; 2]>::try_from(x).unwrap() }).collect::<Vec<_>>()).unwrap(),
        });
    }

    Ok(faces)
}


fn main() {
    let image_path = &args().collect::<Vec<String>>().clone()[1];
    let image = fs::read(image_path).unwrap();

    #[cfg(not(target_arch = "wasm32"))]{
        let now = Instant::now();
        let infer_res = infer(image).unwrap();
        println!("{:?}", now.elapsed());
        println!("{}", infer_res.len());
        for face in infer_res {
            println!("{:?}", face);
        }
    };
    #[cfg(target_arch = "wasm32")]{
        use wasm_bindgen_futures;
        wasm_bindgen_futures::spawn_local(infer(image));
    };
}