use std::env::args;
use std::fmt::format;
use std::path::PathBuf;
use futures::future::join_all;
use ort::tensor::Utf8Data;
use sqlx::SqlitePool;
use sqlx::sqlite::SqliteConnectOptions;
use tokio::fs::{create_dir, File};
use tokio::io::AsyncWriteExt;
use ameba_blog_downloader::data_dir;

#[tokio::main]
async fn main() {
    let dest = match PathBuf::from(args().last().unwrap()).is_dir() {
        true => { PathBuf::from(args().last().unwrap()) }
        false => { unreachable!() }
    };
    let connection_option = SqliteConnectOptions::new().filename(data_dir().join("blog_text.sqlite"));
    let connection = SqlitePool::connect_with(connection_option).await.unwrap();
    let blog_keys = sqlx::query_as("SELECT DISTINCT blog_key FROM blog").fetch_all(&connection).await.unwrap().iter().map(|(v, ): &(String,)| { v.clone() }).collect::<Vec<_>>();
    println!("{:?}", blog_keys);
    let _ = join_all(blog_keys.iter().map(async |key| {
        if !dest.join(key).exists() {
            create_dir(dest.join(key)).await.unwrap();
        }
    }).collect::<Vec<_>>()).await;
    let _ = join_all(blog_keys.iter().map(async |key| {
        let themes = [
            vec!["全員".to_owned()],
            sqlx::query_as("SELECT DISTINCT theme FROM blog WHERE blog_key = ?;").bind(key).fetch_all(&connection).await.unwrap().iter().map(|(x, ): &(String,)| { x.clone() }).collect::<Vec<_>>()
        ].concat();
        join_all(themes.iter().map(async |theme| {
            let mut index_path = match theme.as_str() {
                "全員" => { File::create(dest.join(key).join(theme)) }
                name => { File::create(dest.join(key).join(name)) }
            }.await.unwrap();
            
            index_path.write("".as_utf8_bytes()).await.unwrap();
        })).await;
    }).collect::<Vec<_>>()).await;
}