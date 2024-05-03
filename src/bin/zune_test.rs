use zune_image::image::Image;
fn main() {
    let image =Image::open(r#"C:\Users\tomokazu\すぐ消す\order_test.png"#).unwrap();
    for flatten_frame in image.flatten_frames::<u8>() {
        println!("{:?}", flatten_frame);
    }
}