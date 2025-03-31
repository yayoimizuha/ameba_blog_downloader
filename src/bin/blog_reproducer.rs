use std::env::args;
use std::path::PathBuf;
use futures::future::join_all;
use sqlx::SqlitePool;
use sqlx::sqlite::SqliteConnectOptions;
use tokio::fs::create_dir;
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
        join_all([
            vec!["全員".to_owned()],
            sqlx::query_as("SELECT DISTINCT theme FROM blog WHERE blog_key = ?;").bind(key).fetch_all(&connection).await.unwrap().iter().map(|(x, ): &(String,)| { x.clone() }).collect::<Vec<_>>()
        ].concat().iter().map(async |theme| {
            
        })).await;
    }).collect::<Vec<_>>()).await;
}