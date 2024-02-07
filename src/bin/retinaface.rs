use std::fs;

fn load_model(max_size: i32) {}

fn main() {
    const IMAGE_PATH: &str = "manaka_test.jpg";
    let image_bytes = fs::read(IMAGE_PATH).unwrap();
}