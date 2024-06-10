use std::ops::Deref;
use std::path::Path;
use std::sync::{Arc, Mutex};
use crc32fast::hash;
use futures::future::join_all;
use itertools::Itertools;
use ndarray::{Array, Axis, Zip};
use sqlx::sqlite::SqliteConnectOptions;
use sqlx::SqlitePool;
use tokio::sync;

static SQLITE_DB: sync::OnceCell<Arc<Mutex<SqlitePool>>> = sync::OnceCell::const_new();
const DATA_PATH: &str = r#"D:\helloproject-ai-data"#;

async fn article_cleaner(a: (i64, String), b: (i64, String)) -> Option<(i64, String)> {
    let decrease_new_line = lazy_regex::regex!("(?m)\n{2,}");
    let a_text = decrease_new_line.replace_all(a.1.as_str(), "\n\n").into_owned();
    let b_text = decrease_new_line.replace_all(b.1.as_str(), "\n\n").into_owned();
    let a_hash_list = a_text.split("\n").map(|x| hash(x.as_ref())).collect::<Vec<_>>();
    let b_hash_list = b_text.split("\n").map(|x| hash(x.as_ref())).collect::<Vec<_>>();
    let a_ndarray = Array::from_shape_vec(
        (b_hash_list.len(), a_hash_list.len()), (0..b_hash_list.len()).map(|_| a_hash_list.clone()).flatten().collect()).unwrap();
    let b_ndarray = Array::from_shape_vec(
        (a_hash_list.len(), b_hash_list.len()), (0..a_hash_list.len()).map(|_| b_hash_list.clone()).flatten().collect()).unwrap();
    let compare = Zip::from(&a_ndarray.t()).and(&b_ndarray).map_collect(|&a, &b| (a == b) && a != 0);
    let leave = compare.axis_iter(Axis(0)).map(|x| {
        x.iter().all(|x| !(*x))
    }).collect::<Vec<_>>();
    let return_string = decrease_new_line.replace_all(a_text.split("\n").zip(&leave).filter_map(|(text, cond)| {
        if *cond {
            Some(text)
        } else {
            None
        }
    }).join("\n").as_str(), "\n\n").into_owned();
    if return_string.replace(" ", "").replace("\n", "").replace("\r", "").replace("\t", "").len() == 0 {
        return None;
    }
    Some((a.0, return_string.trim().into()))
}

#[tokio::main]
async fn main() {
    SQLITE_DB.get_or_init(|| async {
        let sqlite_path = Path::new(DATA_PATH).join("blog_text.sqlite");
        let option = SqliteConnectOptions::new().create_if_missing(true).filename(sqlite_path);
        Arc::new(Mutex::new(SqlitePool::connect_with(option).await.unwrap()))
    }).await;
    let themes: Vec<String> = sqlx::query_scalar("SELECT DISTINCT theme from blog;")
        .fetch_all(SQLITE_DB.get().unwrap().lock().unwrap().deref()).await.unwrap();
    for theme in themes {
        let articles = sqlx::query_as("SELECT article_id,article FROM blog WHERE theme = ? ORDER BY date;").bind(&theme)
            .fetch_all(SQLITE_DB.get().unwrap().lock().unwrap().deref()).await.unwrap().iter().map(|(id, text): &(i64, String)| (*id, text.clone())).collect::<Vec<_>>();
        println!("{} {}", theme, articles.len());
        let mut trans = SQLITE_DB.get().unwrap().lock().unwrap().deref().begin().await.unwrap();
        // let last = articles[articles.len() - 2..].to_vec();
        for res in join_all(articles.into_iter().tuple_windows().map(|(a, b)| {
            tokio::spawn(article_cleaner(a, b))
        }).collect::<Vec<_>>()).await {
            let (id, text) = match res.unwrap() {
                None => { continue; }
                Some(x) => x
            };
            sqlx::query("INSERT OR IGNORE INTO processed_blog VALUES(?,NULL,NULL,NULL,NULL);")
                .bind(id).execute(&mut *trans).await.unwrap();
            sqlx::query("UPDATE processed_blog SET article_cleaned = ? WHERE article_id = ?;")
                .bind(text).bind(id).execute(&mut *trans).await.unwrap();
        }
        // let last_res = article_cleaner(last[1].clone(), last[0].clone()).await;
        // sqlx::query("UPDATE processed_blog SET article_cleaned = ? WHERE article_id = ?;")
        //     .bind(last_res.0).bind(last_res.1).execute(&mut *trans).await.unwrap();
        trans.commit().await.unwrap();
    }
}