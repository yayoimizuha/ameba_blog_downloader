use std::env;
use std::path::PathBuf;
use std::sync::Arc;
use futures::future::join_all;
use once_cell::sync::Lazy;
use tokio::fs;
use tokio::sync::Semaphore;
use zune_jpeg::JpegDecoder;
use zune_jpeg::zune_core::bytestream::ZCursor;

const DATA_PATH: Lazy<PathBuf> = Lazy::new(|| {
    match env::var("DATA_PATH") {
        Ok(str) => { PathBuf::from(str) }
        Err(_) => { PathBuf::from(r#"D:\helloproject-ai-data"#) }
    }
});
static SEMAPHORE: Lazy<Arc<Semaphore>> = Lazy::new(|| Arc::new(Semaphore::new(200)));

async fn decode(path: PathBuf) -> Vec<u8> {
    let _permit = SEMAPHORE.acquire().await.unwrap();
    let file_content = fs::read(&path).await.unwrap();
    let mut decoder = JpegDecoder::new(ZCursor::new(&file_content));
    let dec_res = decoder.decode().unwrap();
    println!("{:?}", path);
    dec_res
}

#[tokio::main]
async fn main() {
    let image_dir = DATA_PATH.join("blog_images").read_dir().unwrap().map(|dir| dir.unwrap().path().read_dir().unwrap().into_iter().map(|file| file.unwrap().path())).flatten().collect::<Vec<_>>();
    // println!("{:?}", image_dir);
    let ls = image_dir.into_iter().map(|file| tokio::spawn(decode(file))).collect::<Vec<_>>();
    join_all(ls).await.into_iter();
}