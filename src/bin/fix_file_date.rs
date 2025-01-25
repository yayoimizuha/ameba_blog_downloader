use ameba_blog_downloader::data_dir;
use chrono::DateTime;
use filetime::{set_file_mtime, FileTime};
use itertools::Itertools;
use kdam::{tqdm, BarExt};
use sqlx::sqlite::SqliteConnectOptions;
use sqlx::SqlitePool;
use std::collections::HashMap;
use std::fs::metadata;
use std::ops::Deref;
use std::time::SystemTime;

#[tokio::main]
async fn main() {
    let sqlite_path = data_dir().join("blog_text.sqlite");
    let option = SqliteConnectOptions::new().filename(sqlite_path);
    let pool = SqlitePool::connect_with(option).await.unwrap();
    let articles = sqlx::query_as("SELECT article_id,date FROM blog;").fetch_all(&pool).await.unwrap().iter().map(|(id, date): &(i64, String)| (*id, DateTime::parse_from_rfc3339(date.deref()).unwrap())).collect::<HashMap<_, _>>();

    let dir = data_dir().join("blog_images").read_dir().unwrap();
    // panic!();
    let file_list = dir.filter_map(|x| match x {
        Ok(x) if x.path().is_dir() => {Some(x)}
        _ => {None}
    }).map(|x| x.path().read_dir().unwrap().filter_map(|x| match x {
        Ok(x) if x.path().to_str()?.ends_with(".jpg") => {Some(x)}
        _ => {None}
    }).map(|f| f.path()).collect::<Vec<_>>()).flatten().collect::<Vec<_>>();
    let mut progress_update = tqdm!(total = file_list.len(), desc = "date updating...", animation = "ascii", force_refresh = true, leave = false);
    let _ = file_list
        .iter()
        .map(|x| {
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
        })
        .collect_vec();
}
