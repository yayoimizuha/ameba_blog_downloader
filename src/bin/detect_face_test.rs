use std::env;
use std::io::Read;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use futures::future::join_all;
use once_cell::sync::Lazy;
use tokio::{fs, sync};
use tokio::sync::Semaphore;
use zune_jpeg::JpegDecoder;
use zune_jpeg::zune_core::bytestream::ZCursor;
use ameba_blog_downloader::retinaface::retinaface_common::{ModelKind, RetinaFaceFaceDetector};
use ameba_blog_downloader::retinaface::found_face::FoundFace;
use rand::random;

const DATA_PATH: Lazy<PathBuf> = Lazy::new(|| {
    match env::var("DATA_PATH") {
        Ok(str) => { PathBuf::from(str) }
        Err(_) => { PathBuf::from(r#"D:\helloproject-ai-data"#) }
    }
});
static SEMAPHORE: Lazy<Arc<Semaphore>> = Lazy::new(|| Arc::new(Semaphore::new(200)));
static PREDICTORS: sync::OnceCell<Arc<Vec<Mutex<RetinaFaceFaceDetector>>>> = sync::OnceCell::const_new();

async fn decode(path: PathBuf) -> Option<Vec<FoundFace>> {
    let _permit = SEMAPHORE.acquire().await.unwrap();
    let file_content = fs::read(&path).await.unwrap();
    if file_content.len() == 0 {
        println!("{:?}", path);
        fs::remove_file(path).await.unwrap();
        return None;
    }
    if file_content[..2].iter().zip([255u8, 216u8]).map(|(&a, b)| a == b).any(|x| !x) {
        println!("{:?}", Vec::from(&file_content[..2]));
        println!("{:?}", path);
        fs::remove_file(path).await.unwrap();
        return None;
    }
    let predictor_vec = Arc::clone(&PREDICTORS.get().unwrap());
    let a = &predictor_vec[random::<usize>() % 6].lock().unwrap();
    Some(a.infer(file_content))
    // exit(0);
    // let mut decoder = JpegDecoder::new(ZCursor::new(&file_content));
    // let dec_res = match decoder.decode() {
    //     Ok(x) => {
    //         let predictor_vec = Arc::clone(&PREDICTORS.get().unwrap());
    //         let a = &predictor_vec[random::<usize>() % 6].lock().unwrap();
    //         Some(a.infer(x))
    //     }
    //     Err(x) => {
    //         println!("{:?} : {}", path.as_path(), x.to_string());
    //         None
    //     }
    // };
    // println!("{:?} : {:?}", path, dec_res);
    // dec_res
}

#[tokio::main]
async fn main() {
    PREDICTORS.get_or_init(|| async {
        Arc::new((0..6).map(|_| {
            Mutex::new(RetinaFaceFaceDetector::new(ModelKind::ResNet))
        }).collect::<Vec<_>>())
    }).await;
    let image_dir = DATA_PATH.join("blog_images").read_dir().unwrap().map(|dir| dir.unwrap().path().read_dir().unwrap().into_iter().map(|file| file.unwrap().path())).flatten().collect::<Vec<_>>();
    // println!("{:?}", image_dir);
    let ls = image_dir.into_iter().map(|file| tokio::spawn(decode(file))).collect::<Vec<_>>();
    join_all(ls).await.into_iter();
}