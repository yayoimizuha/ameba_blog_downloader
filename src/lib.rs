extern crate alloc;

use std::env;
use std::path::PathBuf;

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