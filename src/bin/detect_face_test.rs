use std::io::Read;
use std::path::PathBuf;
use std::sync::Arc ;
use once_cell::sync::Lazy;
use std::fs;
use tokio::sync::Semaphore;
use ameba_blog_downloader::retinaface::retinaface_common::{ModelKind, RetinaFaceFaceDetector};
use ameba_blog_downloader::retinaface::found_face::FoundFace;
use ameba_blog_downloader::data_dir;

const DATA_PATH: Lazy<PathBuf> = Lazy::new(|| {
    data_dir()
});
static SEMAPHORE: Lazy<Arc<Semaphore>> = Lazy::new(|| Arc::new(Semaphore::new(1)));
static PREDICTORS: Lazy<RetinaFaceFaceDetector> = Lazy::new(|| RetinaFaceFaceDetector::new(ModelKind::ResNet));

fn decode(path: PathBuf) -> Option<Vec<FoundFace>> {
    // let _permit = SEMAPHORE.acquire().await.unwrap();
    let file_content = fs::read(&path).unwrap();
    if file_content.len() == 0 {
        println!("{:?}", path);
        fs::remove_file(path).unwrap();
        return None;
    }
    if file_content[..2].iter().zip([255u8, 216u8]).map(|(&a, b)| a == b).any(|x| !x) {
        println!("{:?}", Vec::from(&file_content[..2]));
        println!("{:?}", path);
        fs::remove_file(path).unwrap();
        return None;
    }
    {
        // let predictor_vec = Arc::clone(PREDICTORS.get().unwrap());
        // let a = predictor_vec.lock().unwrap();
        println!("{:?}", path);
        Some(PREDICTORS.infer(file_content))
    }
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
    let image_dir = DATA_PATH.join("blog_images").read_dir().unwrap().map(|dir| dir.unwrap().path().read_dir().unwrap().into_iter().map(|file| file.unwrap().path())).flatten().collect::<Vec<_>>();
    // println!("{:?}", image_dir);
    // let ls = image_dir.into_iter().map(|file| tokio::spawn(decode(file))).collect::<Vec<_>>();
    // join_all(ls).await.into_iter();
    let _ = image_dir.into_iter().map(|file| decode(file)).collect::<Vec<_>>();
}