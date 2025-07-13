extern crate alloc;

use std::env;
use std::path::PathBuf;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use bincode::{Decode, Encode};

pub mod retinaface;
pub fn data_dir() -> PathBuf {
    match env::var("DATA_PATH") {
        Ok(str) => { PathBuf::from(str) }
        Err(_) => {
            if cfg!(target_os = "windows") {
                PathBuf::from(r#"D:\helloproject-ai-data"#)
            } else if cfg!(target_os = "linux") {
                PathBuf::from(r#"/media/tomokazu/507E41BA7F31CF88/helloproject-ai-data"#)
            } else {
                unreachable!()
            }
        }
    }
}

pub fn project_dir() -> PathBuf {
    if cfg!(target_os = "windows") {
        PathBuf::from(r#"C:\Users\tomokazu\RustroverProjects\ameba_blog_downloader"#)
    } else if cfg!(target_os = "linux") {
        PathBuf::from(r#"/home/tomokazu/RustroverProjects/ameba_blog_downloader"#)
    } else { unreachable!() }
}
#[derive(Encode, Decode, Debug)]
pub struct Entity {
    pub file_name: String,
    pub file_hash: u128,
    pub embeddings: Vec<f32>,
}

impl PartialEq<Self> for Entity {
    fn eq(&self, other: &Self) -> bool {
        // self.file_name == other.file_name && self.embeddings.iter().zip(&other.embeddings).map(|(a, b)| { a == b }).all(|v| v)
        self.file_hash == other.file_hash && self.file_name == self.file_name
    }
}
impl Hash for Entity {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.file_hash.hash(state);
        self.file_name.hash(state);
    }
}
impl Eq for Entity {}

#[derive(Encode, Decode, Debug)]
pub struct Entities(pub HashSet<Entity>);

