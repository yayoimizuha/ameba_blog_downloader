use std::string::String;
use std::collections::{HashMap, HashSet};
use std::env::current_dir;
use std::fs::{create_dir, File};
use std::io::{BufRead, BufReader};
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;
use chrono::{DateTime, FixedOffset};
use futures::future::join_all;
use once_cell::sync::Lazy;
use reqwest::Client;
use serde_json::Value;
use tokio::{spawn, sync};
use tokio::sync::Semaphore;
use anyhow::Error;
use filetime::{FileTime, set_file_times};
use html5ever:: parse_document;
use html5ever::tendril::TendrilSink;
use markup5ever_rcdom::{RcDom, Handle, NodeData};
use kdam::{Bar, BarExt, tqdm};
use sqlx::SqlitePool;
use sqlx::sqlite::SqliteConnectOptions;
use tokio::io::AsyncWriteExt;

static SEMAPHORE: Lazy<Arc<Semaphore>> = Lazy::new(|| Arc::new(Semaphore::new(200)));
const DATA_PATH: &str = r#"D:\helloproject-ai-data"#;
static SQLITE_DB: sync::OnceCell<Arc<Mutex<SqlitePool>>> = sync::OnceCell::const_new();

static CREATED_USER_DIR: Lazy<Mutex<HashSet<String>>> = Lazy::new(|| Mutex::new(HashSet::new()));

#[derive(Clone, Debug)]
struct PageData {
    article_url: String,
    #[allow(dead_code)]
    comment_api: String,
    last_edit_datetime: DateTime<FixedOffset>,
    theme: String,
    blog_key: String,
    article_id: i64,
    entry_title: String,
}

fn find_init_json(input: String) -> Option<String> {
    match input.find("window.INIT_DATA=") {
        Some(begin) => {
            match input[begin..].find("};") {
                Some(end) => {
                    Some(String::from(
                        &input[begin + "window.INIT_DATA=".len()..begin + end + 1]
                    ))
                }
                None => None,
            }
        }
        None => None
    }
}

async fn get_page_count(client: Client, blog_key: String) -> Option<(String, u64)> {
    let _permit = SEMAPHORE.acquire().await.unwrap();
    match client.get(&format!("https://ameblo.jp/{blog_key}/entrylist.html")).send().await {
        Ok(resp) => {
            let html =resp.text().await.unwrap();
            match serde_json::from_str::<Value>(
                find_init_json(html).unwrap().as_str()
            ) {
                Ok(json) => {
                    match &json["entryState"]["blogPageMap"] {
                        Value::Object(x) => {
                            Some((blog_key,
                                  x.iter().next().unwrap().1["paging"]["max_page"].as_u64().unwrap()
                            ))
                        }
                        _ => unreachable!()
                    }
                }
                Err(_) => None
            }
        }
        Err(_) => None
    }
}

fn create_directory_if_not_exist(theme: &String) {
    let mut unlock = CREATED_USER_DIR.lock().unwrap();
    match unlock.contains(theme) {
        true => {}
        false => {
            match Path::new(DATA_PATH).join("blog_images").join(&theme).exists() {
                true => {}
                false => {
                    create_dir(Path::new(DATA_PATH).join("blog_images").join(&theme)).unwrap();
                    unlock.insert(theme.clone());
                }
            }
        }
    }
}

async fn parse_list_page(client: Client, blog_key: String, page_number: u64, exists: Arc<HashMap<i64, DateTime<FixedOffset>>>, progress: Arc<Mutex<Bar>>) -> Result<Vec<PageData>, Error> {
    let _permit = SEMAPHORE.acquire().await.unwrap();
    let entry_list_url = format!("https://ameblo.jp/{blog_key}/entrylist-{page_number}.html");
    let resp = client.get(&entry_list_url).send().await?.text().await.unwrap();
    let json = serde_json::from_str::<Value>(find_init_json(resp).unwrap().as_str())?;
    progress.lock().unwrap().update(1).unwrap();
    Ok(match json["entryState"]["entryMap"].as_object() {
        None => {
            println!("{}", json);
            println!("{}", entry_list_url);
            return Err(Error::msg(format!("Error occurred at {entry_list_url}")));
        }
        Some(x) => { x }
    }.into_iter().map(|(_, article_info)| {
        let entry_id = article_info["entry_id"].as_i64().unwrap();
        let last_edit_datetime =
            DateTime::parse_from_rfc3339(article_info["last_edit_datetime"].as_str().unwrap()).unwrap();
        match exists.get(&entry_id) {
            None => {}
            Some(date) => {
                if last_edit_datetime == *date {
                    return None;
                };
            }
        }
        let blog_id = article_info["blog_id"].as_i64().unwrap();
        let theme = theme_curator(article_info["theme_name"].as_str().unwrap().to_string(), &blog_key);
        create_directory_if_not_exist(&theme);
        let _article_val = article_info.to_string();
        let entry_title = match article_info["entry_title"].as_str() {
            None => { "".to_string() }
            Some(x) => { x.to_string() }
        };
        let article_url = format!("https://ameblo.jp/{}/entry-{}.html", blog_key, entry_id);
        let comment_api = format!("https://ameblo.jp/_api/blogComments;\
                                amebaId={blog_key};blogId={blog_id};entryId={entry_id};\
                                excludeReplies=false;limit=1;offset=0");
        match article_info["publish_flg"].as_str().unwrap() {
            "open" => {
                Some(PageData {
                    article_url,
                    comment_api,
                    last_edit_datetime,
                    theme,
                    entry_title,
                    blog_key: blog_key.clone(),
                    article_id: entry_id,
                })
            }
            x => {
                eprintln!("{x} at {article_url} in {entry_list_url}");
                None
            }
        }
    }).filter_map(|x| x).collect::<Vec<_>>())
}

fn theme_curator(theme: String, blog_id: &String) -> String {
    let theme_val = match blog_id.as_str() {
        "risa-ogata" => "小片リサ",
        "shimizu--saki" => "清水佐紀",
        "kumai-yurina-blog" => "熊井友理奈",
        "sudou-maasa-blog" => "須藤茉麻",
        "sugaya-risako-blog" => "菅谷梨沙子",
        "miyamotokarin-official" => "宮本佳林",
        "sayumimichishige-blog" => "道重さゆみ",
        "kudo--haruka" => "工藤遥",
        "airisuzuki-officialblog" => "鈴木愛理",
        "angerme-ayakawada" => "和田彩花",
        "miyazaki-yuka-blog" => "宮崎由加",
        "tsugunaga-momoko-blog" => "嗣永桃子",
        "natsuyaki-miyabi-blog" => "夏焼雅",
        "tokunaga-chinami-blog" => "徳永千奈美",
        "tanakareina-blog" => "田中れいな",
        "ozeki-mai-official" => "小関舞",
        "manoerina-official" => "真野恵里菜",
        _ => theme.as_str()
    };
    if theme_val == "梁川 奈々美" {
        "梁川奈々美".to_owned()
    } else {
        theme_val.to_owned()
    }
}

fn html2text(handle: &Handle) -> (String, Vec<String>) {
    let node = handle;
    match node.data {
        NodeData::Text { ref contents } => {
            (contents.borrow().to_string(), vec![])
        }
        NodeData::Element { ref name, ref attrs, .. } => {
            match name.local.to_string().as_str() {
                "br" | "noscript" => ("\n".to_string(), vec![]),
                "img" => {
                    match attrs.borrow().iter().find(|attr| attr.name.local.to_string().as_str() == "class") {
                        None => ("".to_string(), vec![]),
                        Some(attr) => {
                            match attr.value.to_string().as_str() {
                                "PhotoSwipeImage" => {
                                    let image_url = attrs.borrow().iter().find(|attr| attr.name.local.to_string().as_str() == "data-src").unwrap().value.to_string().replace("?caw=800", "?caw=1125");
                                    if !image_url.contains(".jpg?caw") {
                                        ("".to_string(), vec![])
                                    } else {
                                        (format!("-----image-----{}-----",
                                                 attrs.borrow().iter().find(|attr| attr.name.local.to_string().as_str() == "data-image-order").unwrap().value.to_string()),
                                         vec![image_url])
                                    }
                                }
                                _ => ("".to_string(), vec![])
                            }
                        }
                    }
                }
                x => {
                    if x == "div" {
                        match attrs.borrow().iter().find(|attr| attr.name.local.to_string().as_str() == "class") {
                            None => {}
                            Some(attr) => {
                                match attr.value.to_string().as_str() {
                                    "ogpCard_link" => {
                                        return (attrs.borrow().iter().find(|attr|
                                            attr.name.local.to_string().as_str() == "href").unwrap().value.to_string(),
                                                vec![]);
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                    let mut images = vec![];
                    let mut article = "".to_string();
                    let _ = node.children.borrow().iter().map(|child| html2text(child)).map(|(txt, image)| {
                        article = format!("{article}{txt}");
                        images.extend(image)
                    }).collect::<Vec<_>>();
                    if name.local.to_string() == "p" {
                        article += "\n"
                    }
                    (article, images)
                }
            }
        }
        NodeData::Document {} => {
            let mut images = vec![];
            let mut article = "".to_string();
            let _ = node.children.borrow().iter().map(|child| html2text(child)).map(|(txt, image)| {
                article = format!("{article}{txt}");
                images.extend(image)
            }).collect::<Vec<_>>();
            (article, images)
        }
        _ => ("".to_string(), vec![])
    }
}

async fn parse_article_page(client: Client, page_data: PageData, progress: Arc<Mutex<Bar>>) -> Result<(PageData, String, Vec<(String, PathBuf, DateTime<FixedOffset>)>), Error> {
    let _permit = SEMAPHORE.acquire().await.unwrap();
    let resp = client.get(page_data.article_url.clone()).send().await?.text().await.unwrap();
    let json = serde_json::from_str::<Value>(find_init_json(resp).unwrap().as_str())?;
    let (_, &ref entry_main) = json["entryState"]["entryMap"].as_object().unwrap().iter().next().unwrap();
    let dom = parse_document(RcDom::default(), Default::default()).one(entry_main["entry_text"].clone().as_str().unwrap());
    let (text, images) = html2text(&dom.document);
    let images = images.iter().enumerate().map(|(order, url)| (
        url.to_owned(),
        Path::new(DATA_PATH).join("blog_images").join(page_data.theme.clone()).join(
            format!("{}={}={}-{}.jpg", page_data.theme, page_data.blog_key, page_data.article_id, order)),
        page_data.last_edit_datetime
    )).collect::<Vec<_>>();

    progress.lock().unwrap().update(1).unwrap();

    Ok((page_data, text, images))
}

async fn download_file(client: Client, file_path: PathBuf, date: DateTime<FixedOffset>, url: String, id: i64, dl_manager: Arc<Mutex<HashMap<i64, usize>>>, progress: Arc<Mutex<Bar>>) {
    // if file_path.exists() { return; }
    let _permit = SEMAPHORE.acquire().await.unwrap();
    let resp = client.get(url).send().await.unwrap();
    tokio::fs::File::create(file_path.as_path()).await.unwrap().write_all(resp.bytes().await.unwrap().as_ref()).await.unwrap();
    set_file_times(file_path, FileTime::now(), FileTime::from(SystemTime::from(date))).unwrap();
    if {
        let mut lock = dl_manager.lock().unwrap();
        let cnt = lock.get(&id).unwrap().clone();
        if cnt == 1 {
            true
        } else {
            lock.insert(id, cnt - 1);
            false
        }
    } {
        loop {
            match sqlx::query("UPDATE manage SET image_downloaded=1 WHERE article_id = ?;")
                .bind(id)
                .execute(&(|| {
                    SQLITE_DB.get().unwrap().lock().unwrap().clone()
                })()).await {
                Ok(_) => { break; }
                Err(_) => {}
            };
        }
    }
    progress.lock().unwrap().update(1).unwrap();
}

#[tokio::main]
async fn main() {
    SQLITE_DB.get_or_init(|| async {
        let sqlite_path = Path::new(DATA_PATH).join("blog_text.sqlite");
        let option = SqliteConnectOptions::new().create_if_missing(true).filename(sqlite_path);
        let pool = Arc::new(Mutex::new(SqlitePool::connect_with(option).await.unwrap()));
        // let mut conn = pool.lock().unwrap().acquire().await.unwrap();
        sqlx::query("CREATE TABLE IF NOT EXISTS blog (article_id INTEGER PRIMARY KEY,blog_key TEXT,theme TEXT,title TEXT,date TEXT,article TEXT);").execute(pool.lock().unwrap().deref()).await.unwrap();
        sqlx::query("CREATE UNIQUE INDEX IF NOT EXISTS blog_idx ON blog(article_id)").execute(pool.lock().unwrap().deref()).await.unwrap();
        sqlx::query("CREATE TABLE IF NOT EXISTS processed_blog (article_id INTEGER PRIMARY KEY,article_cleaned TEXT,article_overview TEXT,cleaned_embedding BLOB,overview_embedding BLOB);").execute(pool.lock().unwrap().deref()).await.unwrap();
        sqlx::query("CREATE UNIQUE INDEX IF NOT EXISTS processed_blog_idx ON processed_blog(article_id)").execute(pool.lock().unwrap().deref()).await.unwrap();
        sqlx::query("CREATE TABLE IF NOT EXISTS manage (article_id INTEGER PRIMARY KEY,updated_datetime TEXT,image_downloaded INTEGER,comment_downloaded INTEGER,comment_url TEXT);").execute(pool.lock().unwrap().deref()).await.unwrap();
        sqlx::query("CREATE UNIQUE INDEX IF NOT EXISTS manage_idx ON manage(article_id)").execute(pool.lock().unwrap().deref()).await.unwrap();
        sqlx::query("CREATE TABLE IF NOT EXISTS comment (comment_id INTEGER PRIMARY KEY,blog_id INTEGER,user_id TEXT,nickname TEXT,title TEXT,date TEXT,article TEXT);").execute(pool.lock().unwrap().deref()).await.unwrap();
        sqlx::query("CREATE UNIQUE INDEX IF NOT EXISTS comment_idx ON comment (comment_id)").execute(pool.lock().unwrap().deref()).await.unwrap();
        pool
    }).await;

    let blog_names_file: PathBuf = Path::new(&current_dir().unwrap()).join("blog_names.txt");

    let blog_list = BufReader::new(File::open(blog_names_file).unwrap()).lines().map(
        |x| x.unwrap()
    ).collect::<Vec<_>>();
    let reqwest_client = Client::builder()
        // .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0")
        .build().unwrap();

    println!("start count list pages");
    let page_counts = join_all(blog_list.iter().map(|blog_key| {
        spawn(get_page_count(
            reqwest_client.clone(),
            blog_key.into(),
        ))
    })).await.into_iter().map(|x| x.unwrap().unwrap()).collect::<HashMap<_, _>>();

    let exists: Arc<HashMap<_, _>> = Arc::new(sqlx::query_as("SELECT blog.article_id,blog.date FROM blog JOIN manage ON blog.article_id = manage.article_id WHERE manage.image_downloaded <> 0;")
        .fetch_all(SQLITE_DB.get().unwrap().lock().unwrap().deref()).await.unwrap().iter()
        .map(|(a, b): &(i64, String)| (*a, DateTime::parse_from_rfc3339(b.deref()).unwrap())).collect());
    // println!("{:?}", exists);

    let progress_parse_list = Arc::new(Mutex::new(
        tqdm!(total=page_counts.clone().iter().map(|x|*x.1 as usize).sum::<usize>(),
            desc="list page parsing...",animation="ascii",force_refresh=true,leave=false)));
    println!("start parsing list pages");
    let all_pages = join_all(page_counts.into_iter().map(|(blog_key, count)| {
        (1..=count).map(move |x| (blog_key.clone(), x))
    }).flatten().map(|(blog_key, page_num)| {
        spawn({
            let exists = Arc::clone(&exists);
            let progress_clone = Arc::clone(&progress_parse_list);
            parse_list_page(reqwest_client.clone(), blog_key, page_num, exists, progress_clone)
        })
    })).await.into_iter().filter_map(|x| x.unwrap().ok()).flatten().collect::<Vec<_>>();


    let mut trans = SQLITE_DB.get().unwrap().lock().unwrap().deref().begin().await.unwrap();
    // let downloaded_time = Local::now();
    let mut image_download_map: HashMap<i64, Vec<_>> = HashMap::new();

    let progress_parse_article = Arc::new(Mutex::new(tqdm!(total=all_pages.len(),desc="page parsing...",animation="ascii",force_refresh=true,leave=false)));
    let mut progress_update_sql = tqdm!(total=all_pages.len(),desc="sql updating...",animation="ascii",force_refresh=true,leave=false);

    println!("start parsing article pages");
    for (page_data, main_text, a) in join_all(all_pages.into_iter().map(|x| {
        let a = spawn({
            let progress_clone = Arc::clone(&progress_parse_article);
            parse_article_page(reqwest_client.clone(), x, progress_clone)
        }
        );
        a
    })).await.into_iter().filter_map(|x| {
        x.unwrap().ok()
    }) {
        // println!("{:?}", page_data);
        sqlx::query("REPLACE INTO blog VALUES(?, ?, ?, ?, ?, ?)")
            .bind(page_data.article_id)
            .bind(page_data.blog_key)
            .bind(page_data.theme)
            .bind(page_data.entry_title)
            .bind(page_data.last_edit_datetime.to_rfc3339())
            .bind(main_text)
            .execute(&mut *trans).await.unwrap();
        sqlx::query("INSERT OR IGNORE INTO manage VALUES(?, ?, ?, ?, ?)")
            .bind(page_data.article_id)
            .bind(None::<String>)
            // .bind(downloaded_time.to_rfc3339_opts(SecondsFormat::Secs, false))
            .bind(0)
            .bind(0)
            .bind(page_data.comment_api)
            .execute(&mut *trans).await.unwrap();
        if a.is_empty() {
            sqlx::query("UPDATE manage SET image_downloaded=1 WHERE article_id = ?;")
                .bind(page_data.article_id)
                .execute(&mut *trans).await.unwrap();
        } else {
            image_download_map.insert(page_data.article_id, a);
        }
        progress_update_sql.update(1).unwrap();
    }
    for (article_id, _) in image_download_map.clone().iter().filter(|(_, images)| images.is_empty()) {
        sqlx::query("UPDATE manage SET image_downloaded=1 WHERE article_id = ?;")
            .bind(article_id)
            .execute(&mut *trans).await.unwrap();
    };
    trans.commit().await.unwrap();
    let download_counter = Arc::new(Mutex::new(image_download_map.iter().map(|(&article_id, vector)| (article_id, vector.len())).collect::<HashMap<_, _>>()));

    let image_progress = Arc::new(Mutex::new(tqdm!(total=download_counter.lock().unwrap().clone().iter().map(|(_,&size)|size).sum(),desc="image downloading...",animation="ascii",force_refresh=true,leave=false)));


    println!("start downloading images");
    join_all(image_download_map.into_iter().map(|(article_id, images)|
        images.into_iter().map(move |val| (article_id, val))).flatten().collect::<Vec<_>>()
        .into_iter().map(|(id, (url, path, date))| {
        let dl_counter = Arc::clone(&download_counter);
        let progress_clone = Arc::clone(&image_progress);
        spawn(download_file(reqwest_client.clone(), path, date, url, id, dl_counter, progress_clone))
    })).await.into_iter();


    ()
}