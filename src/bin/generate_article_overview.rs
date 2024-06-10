use std::collections::HashMap;
use std::env;
use std::ops::Deref;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use crc32fast::hash;
use futures::future::join_all;
use itertools::Itertools;
use kdam::{Bar, tqdm, BarExt};
use ndarray::{Array, Axis, Zip};
use once_cell::sync::Lazy;
use rand::random;
use reqwest::Client;
use reqwest::header::CONTENT_TYPE;
use serde_json::{json, Value};
use sqlx::sqlite::{SqliteConnectOptions, SqliteQueryResult};
use sqlx::SqlitePool;
use tokio::sync;
use tokio::sync::Semaphore;
use tokio::time::{sleep, sleep_until};

static SQLITE_DB: sync::OnceCell<Arc<Mutex<SqlitePool>>> = sync::OnceCell::const_new();
const DATA_PATH: &str = r#"D:\helloproject-ai-data"#;

static SEMAPHORE: Lazy<Arc<Semaphore>> = Lazy::new(|| Arc::new(Semaphore::new(300)));
static GEMINI_API_KEY: Lazy<String> = Lazy::new(|| env::var("GEMINI_API_KEY").expect("Please set GEMINI_API_KEY."));
// static GEMINI_BEARER_TOKEN: Lazy<String> = Lazy::new(|| env::var("GEMINI_BEARER_TOKEN").expect("Please set GEMINI_BEARER_TOKEN."));

static COUNTER_MAP: Lazy<Arc<Mutex<HashMap<String, (i32, i32)>>>> = Lazy::new(|| Arc::new(Mutex::new(HashMap::new())));
static REQUEST_JSON: Lazy<Value> = Lazy::new(|| json! {
{
    "contents": [
        {
            "role":"USER",
            "parts": [
                {
                    "text": "以下のブログで綴られている主な内容を柔らかい文調で箇条書きで出力してください。あいさつや感想は出力せず、事実のみを出力してください。筆者の名前は「###」です。"
                },
                {
                    "text": "input: "
                },
                {
                    "text": "output: "
                }
            ]
        }
    ],
    "generationConfig": {
        "temperature": 1,
        // "topK": 64,
        "topP": 0.95,
        "maxOutputTokens": 8192,
        "stopSequences": []
    },
    "safetySettings": [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE"
        }
    ]
}
});

async fn article_make_overview(article: String, author: String, article_id: i64, client: Client, progress: Arc<Mutex<Bar>>) {
    let _permit = SEMAPHORE.acquire().await.unwrap();
    {
        let mut bar = progress.lock().unwrap();
        // bar.desc = author;
        bar.update(1).unwrap();
        let mut counter = COUNTER_MAP.lock().unwrap();
        (*counter.get_mut(&author).unwrap()) = (counter.get_mut(&author).unwrap().0 + 1, counter.get_mut(&author).unwrap().1);
        let print_author = "　".repeat(8 - author.chars().count()).to_string() + author.as_str();
        bar.set_description(format!("getting overview... {print_author}: [{:>7}/{:>7}]", counter[&author].0, counter[&author].1));
    }
    if author == "ブログ" || author == "お知らせ" { return; }
    let func_start = Instant::now();
    // let api_url = format!("https://{}-aiplatform.googleapis.com/v1/projects/{}/locations/{}/publishers/google/models/gemini-1.5-flash:generateContent", "us-central1", "gen-lang-client-0379715435", "us-central1");
    let api_url = format!("https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={}", GEMINI_API_KEY.clone());

    let mut request = REQUEST_JSON.clone();
    request["contents"][0]["parts"][0]["text"] = Value::from(request["contents"][0]["parts"][0]["text"].as_str().unwrap().replace("###", author.as_str()));
    request["contents"][0]["parts"][1]["text"] = Value::from(format!("input: {}", article));
    // println!("{}", serde_json::to_string_pretty(&request).unwrap());
    // let mut res = client.post(api_url).json(&request).bearer_auth(GEMINI_BEARER_TOKEN.clone()).send().await.unwrap();
    let mut res = client.post(api_url).json(&request).send().await.unwrap();
    // println!("{}",GEMINI_BEARER_TOKEN.clone());
    // println!("{:?}", res.headers());
    let res_json: Value = res.json().await.unwrap();
    match res_json["candidates"][0]["content"]["parts"][0]["text"].as_str() {
        None => {
            println!("{}", serde_json::to_string_pretty(&res_json.clone()).unwrap());
        }
        Some(text) => {
            // println!("{} {}", author, title);
            loop {
                match sqlx::query("UPDATE processed_blog SET article_overview = ? WHERE article_id = ?;").bind(text.replace(" - ", "").replace("- ", "")).bind(article_id)
                    .execute(&(|| {
                        SQLITE_DB.get().unwrap().lock().unwrap().clone()
                    })()).await {
                    Ok(_) => {
                        break;
                    }
                    Err(err) => {
                        eprintln!("{}", err)
                    }
                };
            }
        }
    }

    let wait = func_start + Duration::from_millis(60 * 1000 + random::<u64>() % (20 * 1000));
    while Instant::now() < wait {
        sleep(Duration::from_millis(100)).await;
        if progress.lock().unwrap().completed() { return; }
    }
    // sleep_until((func_start + Duration::from_secs(60 + random::<u64>() % 10)).into()).await;
    ()
}

#[tokio::main]
async fn main() {
    SQLITE_DB.get_or_init(|| async {
        let sqlite_path = Path::new(DATA_PATH).join("blog_text.sqlite");
        let option = SqliteConnectOptions::new().create_if_missing(true).filename(sqlite_path);
        Arc::new(Mutex::new(SqlitePool::connect_with(option).await.unwrap()))
    }).await;

    let articles = sqlx::query_as("SELECT p.article_id,p.article_cleaned,b.theme FROM processed_blog p JOIN main.blog b on p.article_id = b.article_id WHERE article_cleaned IS NOT NULL  AND article_overview IS NULL ORDER BY b.theme;")
        .fetch_all(SQLITE_DB.get().unwrap().lock().unwrap().deref()).await.unwrap().iter().map(|(id, text, theme): &(i64, String, String)| (*id, text.clone(), theme.clone())).collect::<Vec<_>>();
    sqlx::query_as("SELECT theme,COUNT(theme) FROM blog GROUP BY theme;").fetch_all(SQLITE_DB.get().unwrap().lock().unwrap().deref())
        .await.unwrap().iter().map(|(theme, cnt): &(String, i64)| {
        COUNTER_MAP.lock().unwrap().insert(theme.clone(), (0, *cnt as i32))
    }).count();
    let mut joins = vec![];
    let client = reqwest::Client::new();
    // let sample = articles.into_iter().take(1).collect::<Vec<_>>();
    let progress = Arc::new(Mutex::new(tqdm!(total=articles.len(),desc="get overview...",animation="ascii",force_refresh=true,leave=false,ncols=150u16)));
    for (article_id, article, author) in articles {
        let progress_clone = Arc::clone(&progress);
        joins.push(tokio::spawn(article_make_overview(article, author, article_id, client.clone(), progress_clone)));
    }
    join_all(joins).await.into_iter();
}
