use std::any::Any;
use std::collections::HashMap;
use std::env::args;
use std::fs::{DirEntry, File};
use std::io::Read;
use std::path::PathBuf;
use hdf5_metno::{Dataset, Group};
use itertools::Itertools;
use sha2::{Digest, Sha256};
use ameba_blog_downloader::data_dir;

fn collect_dataset(accumulator: &mut Vec<(Vec<u8>, String)>, group: Group) {
    for inner_group in group.groups().unwrap() {
        collect_dataset(accumulator, inner_group);
    }
    for dataset in group.datasets().unwrap() {
        accumulator.push((dataset.read_raw().unwrap(), dataset.name()));
    }
}

fn collect_file(accumulator: &mut HashMap<String, PathBuf>, path: PathBuf) {
    if path.is_dir() {
        for inner_file in path.read_dir().unwrap() {
            collect_file(accumulator, inner_file.unwrap().path())
        }
    }
    if path.is_file() {
        accumulator.insert(path.file_name().unwrap().to_str().unwrap().to_string(), path);
    }
}
fn main() {
    let face_embeddings = hdf5_metno::File::open(data_dir().join("face_embeddings.hdf5")).unwrap();
    let args = args().collect::<Vec<_>>();
    println!("{:?}", args);
    let [_, model_path, sample_image, source_dir, target_dir] = match args.len() {
        5 => {
            [args[0].clone(), args[1].clone(), args[2].clone(), args[3].clone(), args[4].clone()]
        }
        _ => { unreachable!() }
    };
    let mut model_file = vec![];
    File::open(model_path).unwrap().read_to_end(&mut model_file).unwrap();
    let model_hash = Sha256::digest(model_file).to_ascii_lowercase().into_iter().map(|v| format!("{:02X}", v)).collect::<String>();


    let group = match face_embeddings.groups().unwrap().iter().filter_map(|group| {
        match group.name().contains(model_hash.as_str()) {
            true => { Some(group.clone()) }
            false => { None }
        }
    }).next() {
        None => { panic!("モデルのハッシュが見つかりません。") }
        Some(x) => { x }
    };
    let mut datasets = vec![];
    collect_dataset(&mut datasets, group);
    for dataset in &datasets {
        // println!("{}", dataset.name().rsplitn(2, "/").nth(0).unwrap())
    }
    let mut files = HashMap::new();
    collect_file(&mut files, PathBuf::from(source_dir));
    for file in files {
        println!("{}", file.0)
    }

    println!("{}", datasets.iter().map(|(_, filename)| filename.contains(sample_image.as_str())).any(|v| v));
}