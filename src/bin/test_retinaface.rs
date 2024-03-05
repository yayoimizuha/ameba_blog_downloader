use std::fs;
use ameba_blog_downloader::retinaface::retinaface_common::*;

fn main() {
    let detector = RetinaFaceFaceDetector::new(ModelKind::MobileNet);
    // let image = detector.image_to_array(image::open(r#"C:\Users\tomokazu\RustroverProjects\ameba_blog_downloader\images\horizontal_large.jpg"#).unwrap());
    let res = detector.infer(fs::read(r#"C:\Users\tomokazu\RustroverProjects\ameba_blog_downloader\images\horizontal_large.jpg"#).unwrap());
    println!("{:?}", res);
}