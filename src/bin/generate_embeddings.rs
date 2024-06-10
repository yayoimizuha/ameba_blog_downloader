use std::ops::Deref;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use futures::future::join_all;
use gcp_access_token::json;
use kdam::{Bar, BarExt, tqdm};
use once_cell::sync::Lazy;
use rand::random;
use reqwest::Client;
use serde_json::{json, Value};
use sqlx::sqlite::SqliteConnectOptions;
use sqlx::SqlitePool;
use tokio::sync;
use tokio::sync::Semaphore;
use tokio::time::sleep;

static SQLITE_DB: sync::OnceCell<Arc<Mutex<SqlitePool>>> = sync::OnceCell::const_new();
const DATA_PATH: &str = r#"D:\helloproject-ai-data"#;

static SEMAPHORE: Lazy<Arc<Semaphore>> = Lazy::new(|| Arc::new(Semaphore::new(1000)));
static GCP_TOKEN: Lazy<Arc<Mutex<(String, Instant)>>> = Lazy::new(|| Arc::new(Mutex::new((String::new(), Instant::now()))));
static REGION: &str = "us-central1";
static PROJECT_ID: &str = "gen-lang-client-0379715435";
static REQUEST_JSON: Lazy<Value> = Lazy::new(|| json! {{
  "instances": [
    {
      "task_type": "CLUSTERING",
      // "title": "document title",
      "content": ""
    },
  ]
}
});

async fn update_bearer() {
    // let cred_path = Path::new("./gen-lang-client-0379715435-e27b574bf00a.json");
    // println!("{:?}", cred_path);
    (*GCP_TOKEN.lock().unwrap()).1 = Instant::now();
    let token = gcp_access_token::generator::init_json(&json::parse(include_str!("../../gen-lang-client-0379715435-1627903c0a51.json")).unwrap(), "https://www.googleapis.com/auth/cloud-platform".into()).await.unwrap();
    println!("updated token: {}", json::stringify_pretty(token.clone(), 0).replace("\n", ""));
    (*GCP_TOKEN.lock().unwrap()).0 = token["access_token"].as_str().unwrap().into();
    // GCP_TOKEN.set(Arc::new(Mutex::new((token["access_token"].as_str().unwrap().into(), Instant::now())))).unwrap()
}

async fn get_embedding(article: String, column: String, article_id: i64, client: Client, progress: Arc<Mutex<Bar>>) {
    let _permit = SEMAPHORE.acquire().await.unwrap();
    let token;
    {
        let (tok, exp) = GCP_TOKEN.lock().unwrap().clone();
        // println!("{:?} {:?}", exp, Instant::now());
        if (Instant::now() - exp).as_secs() > 3500 {
            update_bearer().await;
            token = GCP_TOKEN.lock().unwrap().clone().0;
        } else {
            token = tok;
        }
    }
    let func_start = Instant::now();
    // let api_url = format!("https://{}-aiplatform.googleapis.com/v1/projects/{}/locations/{}/publishers/google/models/gemini-1.5-flash:generateContent", "us-central1", "gen-lang-client-0379715435", "us-central1");
    let api_url = format!("https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{REGION}/publishers/google/models/text-embedding-preview-0409:predict");
    // println!("{}", api_url);

    let mut request = REQUEST_JSON.clone();
    request["instances"][0]["content"] = Value::from(article.clone());
    // println!("{}", serde_json::to_string_pretty(&request).unwrap());
    // let mut res = client.post(api_url).json(&request).bearer_auth(GEMINI_BEARER_TOKEN.clone()).send().await.unwrap();
    let res = client.post(api_url).json(&request).bearer_auth(token).send().await.unwrap();
    // println!("{}",GEMINI_BEARER_TOKEN.clone());
    // println!("{:?}", res.headers());
    let res_json: Value = match res.json().await {
        Err(e) => {
            eprintln!("{}", e);
            progress.lock().unwrap().update(1).unwrap();
            return;
        }
        Ok(x) => x
    };
    // println!("{}", serde_json::to_string_pretty(&res_json).unwrap());
    if !res_json["predictions"][0]["embeddings"].as_object().is_some() {
        println!("{} {}", article_id, serde_json::to_string(&res_json).unwrap());
        progress.lock().unwrap().update(1).unwrap();
        return;
    }
    if res_json["predictions"][0]["embeddings"]["statistics"]["truncated"].as_bool().unwrap() {
        println!("{} {}", article_id, serde_json::to_string(&res_json["predictions"][0]["embeddings"]["statistics"]).unwrap());
        progress.lock().unwrap().update(1).unwrap();
        return;
    }
    match res_json["predictions"][0]["embeddings"]["values"].as_array() {
        None => {
            println!("{}", serde_json::to_string_pretty(&res_json.clone()).unwrap());
        }
        Some(vector) => {
            loop {
                // println!("{:?}", vector);
                // println!("{} {}", article_id, serde_json::to_string(&res_json["predictions"][0]["embeddings"]["statistics"]).unwrap());
                let vector_dump = vector.iter().map(|v| (v.as_f64().unwrap() as f32).to_le_bytes()).flatten().collect::<Vec<_>>();
                match sqlx::query(format!("UPDATE processed_blog SET {column} = ? WHERE article_id = ?;").as_str())
                    .bind(vector_dump).bind(article_id)
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
    {
        let mut bar = progress.lock().unwrap();
        bar.update(1).unwrap();
    }
    let wait = func_start + Duration::from_millis(45 * 1000 + random::<u64>() % (1000 * 30));
    while Instant::now() < wait {
        sleep(Duration::from_millis(100)).await;
        if progress.lock().unwrap().completed() { return; }
    }
    ()
}

static TARGET: &str = "overview";

#[tokio::main]
async fn main() {
    SQLITE_DB.get_or_init(|| async {
        let sqlite_path = Path::new(DATA_PATH).join("blog_text.sqlite");
        let option = SqliteConnectOptions::new().create_if_missing(true).filename(sqlite_path);
        Arc::new(Mutex::new(SqlitePool::connect_with(option).await.unwrap()))
    }).await;
    update_bearer().await;

    let articles = sqlx::query_as(format!("SELECT article_id,article_{TARGET} FROM processed_blog WHERE article_{TARGET} IS NOT NULL AND {TARGET}_embedding IS NULL;").as_str())
        .fetch_all(SQLITE_DB.get().unwrap().lock().unwrap().deref()).await.unwrap().iter().map(|(id, text): &(i64, String)| (*id, text.clone())).collect::<Vec<_>>();
    let mut joins = vec![];
    let client = reqwest::Client::new();
    // let sample = articles.into_iter().take(1).collect::<Vec<_>>();
    let progress = Arc::new(Mutex::new(tqdm!(total=articles.len(),desc="get embedding...",animation="ascii",force_refresh=true,leave=false)));

    for (article_id, article) in articles {
        let progress_clone = Arc::clone(&progress);
        joins.push(tokio::spawn(get_embedding(article, format!("{TARGET}_embedding"), article_id, client.clone(), progress_clone)));
    }
    join_all(joins).await.into_iter();
}
