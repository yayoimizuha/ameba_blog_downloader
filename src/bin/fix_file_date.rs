use std::collections::HashMap;
use std::fs::metadata;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::time::SystemTime;
use chrono::DateTime;
use filetime::{FileTime, set_file_mtime};
use itertools::Itertools;
use kdam::{BarExt, tqdm};
use sqlx::sqlite::SqliteConnectOptions;
use sqlx::SqlitePool;

const DATA_PATH: &str = r#"D:\helloproject-ai-data"#;

#[tokio::main]
async fn main() {
    let sqlite_path = Path::new(DATA_PATH).join("blog_text.sqlite");
    let option = SqliteConnectOptions::new().filename(sqlite_path);
    let pool = SqlitePool::connect_with(option).await.unwrap();
    let articles = sqlx::query_as("SELECT article_id,date FROM blog;").fetch_all(&pool).await.unwrap()
        .iter().map(|(id, date): &(i64, String)| {
        (*id, DateTime::parse_from_rfc3339(date.deref()).unwrap())
    }).collect::<HashMap<_, _>>();

    let dir = PathBuf::from(DATA_PATH).join("blog_images").read_dir().unwrap();
    let file_list = dir.map(|x| x.unwrap().path().read_dir().unwrap().map(|f| {
        f.unwrap().path()
    }).collect::<Vec<_>>()).flatten().collect::<Vec<_>>();
    let mut progress_update = tqdm!(total=file_list.len(),desc="date updating...",animation="ascii",force_refresh=true,leave=false);
    let _ = file_list.iter().map(|x| {
        let key = x.file_name().unwrap().to_str().unwrap().split("=").nth(2).unwrap().split("-").nth(0).unwrap().parse::<i64>().unwrap();
        match articles.get(&key) {
            None => {}
            Some(date) => {
                let filetime = FileTime::from(SystemTime::from(*date));
                if filetime != FileTime::from_last_modification_time(&metadata(x).unwrap()) {
                    set_file_mtime(x, filetime).unwrap();
                }
            }
        }
        progress_update.update(1).unwrap();
    }).collect_vec();
}