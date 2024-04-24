use std::path::Path;
use std::fs;
use zune_jpeg::JpegDecoder;

const DATA_PATH: &str = r#"D:\helloproject-ai-data"#;


fn main() {
    let image_dir = Path::new(DATA_PATH).join("blog_images").read_dir().unwrap().map(|dir|dir.unwrap().path().read_dir().unwrap().into_iter().map(|file|file.unwrap().path())).flatten().collect::<Vec<_>>();
    // println!("{:?}", image_dir);
    let _ = image_dir.iter().map(|file| {
        let file_content = fs::read(file).unwrap();
        let mut decoder =JpegDecoder::new(&*file_content);
        decoder.decode().unwrap();
    }).collect::<Vec<_>>();
}