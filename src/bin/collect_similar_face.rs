use ameba_blog_downloader::{data_dir, Entities, Entity};
use bincode::config;
use std::env;
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;
use std::process::Command;
const EMBEDDING_SIZE: usize = 512;
const TAKE_NTH: usize = 400;
fn main() {
    let mut cache = Vec::new();
    File::open(data_dir().join("embeddings_cache").join("model_82F2DE1C9D7C7F086BA8AD6518B31C3D761084A3F7849E9899858620FAA82303.bin")).unwrap().read_to_end(&mut cache).unwrap();
    let data: Entities = bincode::decode_from_slice(cache.as_slice(), config::standard()).unwrap().0;
    let dest_dir = PathBuf::from(env::args().nth(1).unwrap());
    let target_files = env::args().skip(2).collect::<Vec<_>>();
    println!("{:?}", target_files);
    let target_embeddings: Vec<[f32; EMBEDDING_SIZE]> = target_files.iter().map(|file| {
        let file = file.to_owned();
        data.0.iter().find(|entity| entity.file_name.ends_with(file.as_str())).unwrap().embeddings.clone().try_into().expect("Vec の長さが違います")
    }).collect::<Vec<_>>();

    let mut similarity: Vec<(Entity, f32)> = Vec::new();

    data.0.into_iter().for_each(|each_files| {
        let mut sum = 1f32;
        target_embeddings.iter().for_each(|&target| {
            let inner_product = each_files.embeddings.iter().take(EMBEDDING_SIZE).zip(target.iter()).map(|(a, b)| {
                a * b
            }).sum::<f32>();
            let norm_a = each_files.embeddings.iter().map(|a| {
                a * a
            }).sum::<f32>();
            let norm_b = target.iter().map(|a| {
                a * a
            }).sum::<f32>();

            let cosine_similarity = inner_product / (norm_a.sqrt() * norm_b.sqrt());
            sum *= cosine_similarity;
        });
        similarity.push((each_files, sum / target_embeddings.len() as f32));
    });
    similarity.select_nth_unstable_by(TAKE_NTH, |a, b| b.1.partial_cmp(&a.1).unwrap());
    similarity.iter().take(TAKE_NTH).for_each(|(entity, score)| {
        println!("{:?} : {}", entity.file_name, score);
    });
    // copy to dest_dir
    let mut exiftool_processes = Vec::new();

    std::fs::create_dir_all(dest_dir.clone()).unwrap();
    similarity.into_iter().take(TAKE_NTH).enumerate().for_each(|(order, (entity, score))| {
        let src = PathBuf::from(entity.file_name.clone());
        let dest_path = dest_dir.join(src.file_name().unwrap().to_str().unwrap());
        std::fs::copy(&src, &dest_path).unwrap();

        exiftool_processes.push(Command::new("exiftool")
            .arg("-overwrite_original")
            .arg(format!("-UserComment={}", format!("{:.6}", score)))
            .arg(dest_path.to_str().unwrap())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .expect("failed to execute exiftool"));
    });
    for mut process in exiftool_processes {
        process.wait().expect("failed to wait on exiftool");
    }
}
