use std::fs;
use std::fs::File;
use wgpu::{Backend};
use zune_jpeg::JpegDecoder;
use zune_jpeg::zune_core::bytestream::ZCursor;

fn main() {
    let file_bin = fs::read(r#"C:\Users\tomokazu\friends-4385686.jpg"#).unwrap();
    let jpeg_data = JpegDecoder::new(ZCursor::new(file_bin)).decode().unwrap();
}