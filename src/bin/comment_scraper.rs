use std::collections::HashMap;
use std::ops::Deref;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use chrono::{DateTime, FixedOffset, Local};
use futures::executor::block_on;
use futures::future::join_all;
use kdam::{Bar, BarExt, tqdm};
use once_cell::sync::Lazy;
use rand::random;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sqlx::sqlite::SqliteConnectOptions;
use sqlx::SqlitePool;
use tokio::sync;
use tokio::sync::Semaphore;
use tokio::time::sleep;


#[derive(Serialize, Deserialize, Debug)]
struct CommentAuthor {
    ameba_id: String,
    blog_id: i64,
    nickname: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct CommentElement {
    comment_id: i64,
    blog_id: i64,
    entry_id: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    comment_author: Option<CommentAuthor>,
    comment_name: String,
    comment_url: String,
    comment_title: String,
    comment_add_flg: String,
    ins_datetime: DateTime<FixedOffset>,
    upd_datetime: DateTime<FixedOffset>,
    record_status: String,
    public_comment_flg: String,
    comment_text: String,
    reply_count: i64,
}

#[derive(Serialize, Deserialize, Debug)]
struct Paging {
    prev: i64,
    next: i64,
    limit: i64,
    total_count: i64,
    max_page: i64,
    current_page: i64,
    next_page: i64,
    prev_page: i64,
    order: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct CommentsJson {
    #[serde(rename = "commentMap")]
    comment_map: HashMap<i64, CommentElement>,
    #[serde(rename = "commentIds")]
    comment_ids: Vec<i64>,
    paging: Paging,
}

static SQLITE_DB: sync::OnceCell<Arc<Mutex<SqlitePool>>> = sync::OnceCell::const_new();
const DATA_PATH: &str = r#"D:\helloproject-ai-data"#;
static SEMAPHORE: Lazy<Arc<Semaphore>> = Lazy::new(|| Arc::new(Semaphore::new(20)));

async fn download_comments(client: Client, comment_url: String, article_id: i64, download_progress: Arc<Mutex<Bar>>) {
    let _permit = SEMAPHORE.acquire().await.unwrap();
    let func_start = Instant::now();
    let head = client.get(&comment_url).send().await.unwrap().text().await.unwrap();
    match serde_json::from_str::<Value>(head.as_str()).unwrap().get("commentMap") {
        None => {
            eprintln!("no data at {comment_url}");
            download_progress.lock().unwrap().update(1).unwrap();
            return;
        }
        Some(_) => {}
    }
    let head_json = match serde_json::from_str::<CommentsJson>(head.as_str()) {
        Ok(x) => { x }
        Err(err) => {
            eprintln!("{} at {}", err, comment_url);
            // eprintln!("{}", head);
            download_progress.lock().unwrap().update(1).unwrap();
            return;
        }
    };
    let mut trans;
    {
        let sqlite_db = SQLITE_DB.get().unwrap().lock().unwrap();
        trans = block_on(sqlite_db.deref().begin()).unwrap();
    }
    let main_query = client.get(comment_url.replace("limit=1", format!("limit={}", head_json.paging.total_count).as_str()))
        .send().await.unwrap().text().await.unwrap();

    let main_query_json = match serde_json::from_str::<CommentsJson>(main_query.as_str()) {
        Ok(x) => { x }
        Err(err) => {
            eprintln!("{} at {}", err, comment_url);
            // eprintln!("{}", main_query);
            download_progress.lock().unwrap().update(1).unwrap();
            return;
        }
    };
    for (comment_id, comment_element) in main_query_json.comment_map {
        // println!("{:?}", comment_element);
        let (user_id, nickname) = {
            match comment_element.comment_author {
                None => { (None::<i64>, Some(comment_element.comment_name)) }
                Some(x) => { (Some(x.blog_id), Some(x.nickname)) }
            }
        };
        sqlx::query("REPLACE INTO comment VALUES(?,?,?,?,?,?,?);")
            .bind(comment_id)
            .bind(comment_element.blog_id)
            .bind(user_id)
            .bind(nickname.unwrap())
            .bind(comment_element.comment_title)
            .bind(comment_element.upd_datetime)
            .bind(comment_element.comment_text)
            .execute(&mut *trans).await.unwrap();
    }
    sqlx::query("UPDATE manage SET comment_downloaded = ?,updated_datetime = ? WHERE article_id = ?;")
        .bind(true)
        .bind(Local::now().fixed_offset())
        .bind(article_id).execute(&mut *trans).await.unwrap();
    trans.commit().await.unwrap();
    download_progress.lock().unwrap().update(1).unwrap();
    let wait = func_start + Duration::from_millis(5 * 1000 + random::<u64>() % (1000 * 1));
    while Instant::now() < wait {
        sleep(Duration::from_millis(100)).await;
        if download_progress.lock().unwrap().completed() { return; }
    }
}

#[tokio::main]
async fn main() {
    SQLITE_DB.get_or_init(|| async {
        let sqlite_path = Path::new(DATA_PATH).join("blog_text.sqlite");
        let option = SqliteConnectOptions::new().create_if_missing(true).filename(sqlite_path);
        Arc::new(Mutex::new(SqlitePool::connect_with(option).await.unwrap()))
    }).await;
    let client = Client::new();
    // let res = client.get("https://ameblo.jp/_api/blogComments;amebaId=juicejuice-official;blogId=10039630379;entryId=11618721121;excludeReplies=false;limit=1;offset=0").send().await.unwrap();
    // let deserialized = serde_json::from_str::<CommentsJson>(res.text().await.unwrap().as_str()).unwrap();
    // println!("{:?}", deserialized);
    let comment_urls = sqlx::query_as("SELECT m.article_id, m.updated_datetime, m.comment_downloaded, m.comment_url, b.date FROM manage m JOIN blog b on m.article_id = b.article_id")
        .fetch_all(SQLITE_DB.get().unwrap().lock().unwrap().deref()).await.unwrap().iter().map(|(article_id, updated_datetime, comment_downloaded, comment_url, page_datetime): &(i64, String, bool, String, String)| {
        (*article_id, DateTime::parse_from_rfc3339(&*updated_datetime.clone()), *comment_downloaded, comment_url.clone(), DateTime::parse_from_rfc3339(&*page_datetime.clone()).unwrap())
    }).collect::<Vec<_>>();
    // println!("{:?}", comment_urls[0]);
    let download_urls = comment_urls.into_iter().filter_map(|x| {
        let (_, updated_datetime, comment_downloaded, _, page_datetime) = x.clone();
        if comment_downloaded {
            match updated_datetime {
                Ok(d) => {
                    if (Local::now().fixed_offset() - d).num_days() > 30 && (Local::now().fixed_offset() - page_datetime).num_days() < 180 { Some(x) } else { None }
                }
                Err(_) => { Some(x) }
            }
        } else {
            Some(x)
        }
    }).collect::<Vec<_>>();
    let download_progress = Arc::new(Mutex::new(tqdm!(total=download_urls.len(),desc="comment downloading...",animation="ascii",force_refresh=true,leave=false)));
    // for (article_id, _, _, comment_url, _) in download_urls {
    //     println!("{}", comment_url);
    // let cloned_progress = Arc::new(&download_progress);
    //
    // };
    join_all(download_urls.into_iter().map(|(article_id, _, _, comment_url, _)| {
        let cloned_progress = Arc::clone(&download_progress);
        tokio::spawn(download_comments(client.clone(), comment_url, article_id, cloned_progress))
    })).await;
    ()
}