use std::{f32};
use std::ops::{Add, Div, Mul};
use std::time::Instant;
use image::DynamicImage;
use itertools::{enumerate, iproduct, Itertools};
use ndarray::{arr1, arr2, Array, array, Array2, Array4, Axis, concatenate, Ix1, Ix2, s};
use ort::{inputs, Session, Value};


use super::found_face::FoundFace;

// const ONNX_PATH: &str = r#"C:\Users\tomokazu\RustroverProjects\ameba_blog_downloader\src\bin\mobilenet_retinaface.onnx"#;


pub fn transform(image: DynamicImage/*, _max_size: usize*/) -> Array4<f32> {
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
        output_image[[0usize, 0, y as usize, x as usize]] = r;
        output_image[[0usize, 1, y as usize, x as usize]] = g;
        output_image[[0usize, 2, y as usize, x as usize]] = b;
    }
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
                let s_kx = min_size as f32 / image_size[1] as f32;
                let s_ky = min_size as f32 / image_size[0] as f32;
                let dense_cx = [j as f32 + 0.5].iter().map(|x| x * steps[k] as f32 / image_size[1] as f32).collect::<Vec<_>>();
                let dense_cy = [i as f32 + 0.5].iter().map(|y| y * steps[k] as f32 / image_size[0] as f32).collect::<Vec<_>>();
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

fn nms_impl(boxes: Array<f32, Ix2>, scores: Array<f32, Ix1>, nms_threshold: f32) -> Vec<usize> {
    let x1 = boxes.slice(s![..,0]).to_owned();
    let y1 = boxes.slice(s![..,1]).to_owned();
    let x2 = boxes.slice(s![..,2]).to_owned();
    let y2 = boxes.slice(s![..,3]).to_owned();

    let areas = (x2.clone() - x1.clone()).add(1.) * (y2.clone() - y1.clone()).add(1.);
    let mut order = scores.iter().enumerate().sorted_by(|a, b| b.1.partial_cmp(a.1).unwrap()).map(|x| x.0).collect::<Vec<_>>();

    let mut keep = vec![];

    let np_maximum = |_x1: f32, _x2: Array<f32, Ix1>| -> Array<f32, Ix1> {
        _x2.mapv(|_x2_val| if _x2_val > _x1 { _x2_val } else { _x1 })
    };
    let np_minimum = |_x1: f32, _x2: Array<f32, Ix1>| -> Array<f32, Ix1> {
        _x2.mapv(|_x2_val| if _x2_val < _x1 { _x2_val } else { _x1 })
    };
    while order.len() > 0 {
        let i = order[0];
        keep.push(i);

        let xx1 = np_maximum(x1[[i]], x1.select(Axis(0), &order[1..]));
        let yy1 = np_maximum(y1[[i]], y1.select(Axis(0), &order[1..]));
        let xx2 = np_minimum(x2[[i]], x2.select(Axis(0), &order[1..]));
        let yy2 = np_minimum(y2[[i]], y2.select(Axis(0), &order[1..]));

        let w = np_maximum(0.0, (xx2 - xx1).add(1.));
        let h = np_maximum(0.0, (yy2 - yy1).add(1.));

        let inter = w * h;
        let ovr = inter.clone() / (areas.select(Axis(0), &order[1..]).add(areas[[i]]) - inter.clone());

        let indices = ovr.iter().enumerate().filter(|(_, val)| val < &&nms_threshold).map(|(order, _)| order).collect::<Vec<_>>();

        order = arr1(&*order).select(Axis(0), &*indices.iter().map(|x| x + 1).collect::<Vec<_>>()).to_vec();
    }
    keep
}


pub fn infer(session: &Session, image_bytes: Vec<u8>) -> Result<Vec<FoundFace>, String> {
    const _MAX_SIZE: usize = 640;

    let confidence_threshold = 0.02;
    let nms_threshold = 0.4;
    let vis_threshold = 0.6;
    // let keep_top_k = 750;
    let top_k = 5000;
    let variance = [0.1, 0.2];

    let image;
    match image::load_from_memory(image_bytes.as_slice()) {
        Ok(i) => { image = i }
        Err(err) => {
            eprintln!("Error while loading image: {}", err);
            return Err(err.to_string());
        }
    };
    let raw_image = transform(image);
    println!("{:?}", raw_image.dim());

    let binding = raw_image.to_owned();
    let input_shape = binding.shape();
    let onnx_input = inputs!["input"=>raw_image.view()].unwrap();
    let transformed_size = array![input_shape[3], input_shape[2]].to_owned();

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

    let mut boxes = decode(loc.slice(s![0,..,..]).to_owned(), prior_box.clone(), variance);
    boxes = boxes * scale_bboxes;

    let mut scores = confidence.slice(s![0,..,1]).to_owned() as Array<f32, Ix1>;
    let mut landmarks = decode_landmark(landmark.slice(s![0,..,..]).to_owned(), prior_box.clone(), variance);
    landmarks = landmarks * scale_landmarks;

    let indices = scores.iter().enumerate().filter(|(_, val)| val > &&confidence_threshold).map(|(order, _)| order).collect::<Vec<_>>();
    boxes = boxes.select(Axis(0), &*indices);
    landmarks = landmarks.select(Axis(0), &*indices);
    scores = scores.select(Axis(0), &*indices);

    let mut order = scores.clone().iter().enumerate().sorted_by(|a, b| b.1.partial_cmp(a.1).unwrap()).map(|x| x.0).collect::<Vec<_>>();
    if order.len() > top_k {
        order = order[..top_k].to_vec()
    }
    boxes = boxes.select(Axis(0), &*order);
    landmarks = landmarks.select(Axis(0), &*order);
    scores = scores.select(Axis(0), &*order);


    let keep = nms_impl(boxes.clone(), scores.clone(), nms_threshold);


    let boxes = boxes.select(Axis(0), &*keep);
    let scores = scores.select(Axis(0), &*keep);
    let landmarks = landmarks.select(Axis(0), &*keep);

    let vis_score_keep = scores.iter().enumerate().filter(|x| x.1 > &vis_threshold).map(|x| x.0).collect::<Vec<_>>();


    let mut faces = vec![];
    for index in vis_score_keep {
        faces.push(FoundFace {
            bbox: <[f32; 4]>::try_from(boxes.slice(s![index,..]).to_vec()).unwrap(),
            score: *scores.get(index).unwrap(),
            landmarks: <[[f32; 2]; 5]>::try_from(landmarks.slice(s![index,..]).to_vec().chunks_exact(2).map(|x| { <[f32; 2]>::try_from(x).unwrap() }).collect::<Vec<_>>()).unwrap(),
        });
        // print!("{}\t", boxes.slice(s![index,..]));
        // print!("{}\t", scores.slice(s![index]));
        // println!("{}", landmarks.slice(s![index,..]));
    }
    // println!("{:?}", faces);

    Ok(faces)
}

//
// fn main() {
//     let image_path = &args().collect::<Vec<String>>().clone()[1];
//     let image = fs::read(image_path).unwrap();
//
//     #[cfg(not(target_arch = "wasm32"))]{
//         let now = Instant::now();
//         let infer_res = infer(image).unwrap();
//         println!("{:?}", now.elapsed());
//         println!("{}", infer_res.len());
//         for face in infer_res {
//             println!("{:?}", face);
//         }
//     };
//     #[cfg(target_arch = "wasm32")]{
//         use wasm_bindgen_futures;
//         wasm_bindgen_futures::spawn_local(infer(image));
//     };
// }