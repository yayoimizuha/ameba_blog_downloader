use std::clone::Clone;
use std::fs::create_dir;
use std::path::Path;
use std::str::FromStr;
use std::sync::Arc;
use std::time::SystemTime;
use chrono::{DateTime, FixedOffset, Utc};
use futures::future;
use html5ever::tree_builder::TreeSink;
use regex::Regex;
use reqwest::Client;
use serde_json::Value;
use tokio::{time, spawn};
use tokio::fs as async_fs;
use tokio::sync::Semaphore;
use scraper::{Html, Selector};
use tokio::io::AsyncWriteExt;
use filetime::{FileTime, set_file_times};
// use kdam::{Bar, BarExt};

#[allow(dead_code)]
const NAME_ALL: &[&str] = &["angerme-ss-shin", "angerme-amerika", "angerme-new", "juicejuice-official",
    "tsubaki-factory", "morningmusume-9ki", "morningmusume-10ki", "mm-12ki", "morningm-13ki",
    "morningmusume15ki", "morningmusume16ki", "beyooooonds-rfro", "beyooooonds-chicatetsu",
    "beyooooonds", "ocha-norma", "countrygirls", "risa-ogata", "kumai-yurina-blog",
    "sudou-maasa-blog", "sugaya-risako-blog", "miyamotokarin-official", "kobushi-factory",
    "sayumimichishige-blog", "kudo--haruka", "airisuzuki-officialblog", "angerme-ayakawada",
    "miyazaki-yuka-blog", "tsugunaga-momoko-blog", "tokunaga-chinami-blog", "c-ute-official",
    "tanakareina-blog"];

#[allow(dead_code)]
const TEST_NAME: &[&str] = &["airisuzuki-officialblog"];


const NAMES: &[&str] = TEST_NAME;


async fn async_wait(t: u64) { time::sleep(time::Duration::from_millis(t)).await }

#[derive(Clone, Debug)]
struct PageData {
    blog_page: String,
    #[allow(dead_code)]
    comment_api: String,
    last_edit_datetime: DateTime<FixedOffset>,
    theme: String,
    blog_account: String,
    article_id: u64,
}

#[derive(Clone, Debug)]
struct ImageData {
    page_data: PageData,
    filename: String,
    theme: String,
    url: String,
    date: DateTime<Utc>,
}


fn html_to_text(_html: Value, json: Value, page_data: PageData) -> (Vec<ImageData>, String) {
    // println!("{:?}", page_data);
    let mut return_val = vec![];
    // if page_data.blog_page != "https://ameblo.jp/airisuzuki-officialblog/entry-12340462803.html" {
    //     return (vec![], "".to_string());
    // }
    // println!("{}", page_data.blog_page);
    let mut html = Html::parse_document(&*_html.as_str().unwrap().replace("<br>", "\n"));
    let last_edit_date = DateTime::<Utc>::from_str(json["last_edit_datetime"].as_str().unwrap()).unwrap();
    // println!("{:?}", html.html());
    // let emoji: Selector = Selector::parse("img.PhotoSwipeImage[data-src]").unwrap();

    let emoji_selector = Selector::parse("img.emoji").unwrap();
    let emojis: Vec<_> = html.select(&emoji_selector).map(|x| x.id()).collect();
    for emoji in &emojis {
        html.remove_from_parent(&emoji);
    }

    let no_script: Selector = Selector::parse("noscript").unwrap();
    let no_scripts: Vec<_> = html.select(&no_script).map(|x| x.id()).collect();
    for noscript in no_scripts {
        html.remove_from_parent(&noscript);
    }


    let image_selector = Selector::parse("img[class=PhotoSwipeImage]").unwrap();
    let images: Vec<_> = html.select(&image_selector).map(|x| x.clone()).collect();
    // println!("{}", html.html());

    for (order, image) in images.iter().enumerate() {
        // println!("{}", image.html());
        // println!("{}", image.value().attr("data-src").unwrap());
        return_val.push(ImageData {
            page_data: page_data.clone(),
            filename: format!("{}={}={}-{}.jpg", page_data.theme, page_data.blog_account, page_data.article_id, order + 1),
            url: image.value().attr("data-src").unwrap().to_string().replace("?caw=800", "?caw=1125"),
            date: last_edit_date,
            theme: page_data.theme.clone(),
        })
    }
    // for text in texts {
    //     println!("{}", text)
    // }
    let as_text = Selector::parse("*").unwrap();
    let texts = html.select(&as_text).next().unwrap().text().map(|x| String::from_str(x).unwrap()).collect::<Vec<_>>();
    (return_val, texts.join("\n"))
}


async fn get_page_count(client: Client, name: &str, page_count: Regex) -> u64 {
    let mut page_nums: Option<u64> = None;
    // let ASYNC_WAIT = |t: u64| async move { time::sleep(time::Duration::from_millis(t)).await };
    while page_nums.is_none() {
        match client
            .get(&format!("https://ameblo.jp/{name}/entrylist.html"))
            .send()
            .await
        {
            Ok(response) => {
                let text = response.text().await.unwrap();
                match page_count.captures(&*text).unwrap().get(0).ok_or(&text) {
                    Ok(matched_text) => {
                        let mut json_str = matched_text.as_str().to_string();
                        json_str.pop();
                        json_str = json_str.replacen("<script>window.INIT_DATA=", "", 1);
                        match serde_json::from_str::<Value>(&json_str) {
                            Ok(json) => {
                                let _ = json["entryState"]["blogPageMap"]
                                    .as_object()
                                    .unwrap()
                                    .iter()
                                    .map(|(_, x)| {
                                        page_nums = Option::from(
                                            x["paging"]["max_page"].as_u64().unwrap(),
                                        );
                                    })
                                    .collect::<Vec<_>>();
                            }
                            Err(err) => {
                                eprintln!("Failed to parse json string: {}", err);
                                async_wait(1500).await;
                            }
                        }
                    }
                    Err(err) => {
                        eprintln!("Failed to get json string: {}", err);
                        async_wait(1500).await;
                    }
                }
            }
            Err(err) => {
                eprintln!("Failed to get entry_list page.: {}", err);
                async_wait(1500).await;
            }
        }
        async_wait(1500).await;
    }
    page_nums.unwrap()
    //page_count
}

async fn parse_list_page(client: Client, blog_name: &str, page: u64, matcher: Regex, semaphore: Arc<Semaphore>) -> Vec<PageData> {
    let mut page_urls: Option<Vec<PageData>> = None;
    let _permit = semaphore.acquire().await.unwrap();
    while page_urls.is_none() {
        match client.get(&format!("https://ameblo.jp/{blog_name}/entrylist-{page}.html"))
            .send().await {
            Ok(resp) => {
                let resp_text = resp.text().await.unwrap().as_str().to_string();
                match matcher.captures(&*resp_text).unwrap().get(0).ok_or(&resp_text) {
                    Ok(matched_text) => {
                        let mut json_str = matched_text.as_str().to_string();
                        json_str.pop();
                        json_str = json_str.replacen("<script>window.INIT_DATA=", "", 1);
                        match serde_json::from_str::<Value>(&json_str) {
                            Ok(json) => {
                                // let vec = vec![];
                                page_urls = Some(vec![]);
                                json["entryState"]["entryMap"].as_object().unwrap().iter().for_each(|(_, x)| {
                                    // println!("{}", x);
                                    if x["publish_flg"] == "open" {
                                        let article_url = format!("https://ameblo.jp/{}/entry-{}.html", blog_name, x["entry_id"]);
                                        // println!("{}", article_url);
                                        match page_urls.as_mut() {
                                            None => unreachable!(),
                                            Some(list) => {
                                                let page_data = PageData {
                                                    blog_page: article_url.clone(),
                                                    comment_api: ["https://ameblo.jp/_api/blogComments".to_string(), format!("amebaId={blog_name}"),
                                                        format!("blogId={}", x["blog_id"]), format!("entryId={}", x["entry_id"]),
                                                        "excludeReplies=false".to_string(), "limit=1".to_string(), "offset=0".to_string()].join(";"),
                                                    last_edit_datetime: DateTime::parse_from_rfc3339(x["last_edit_datetime"].as_str().unwrap()).unwrap(),
                                                    theme: theme_curator(x["theme_name"].as_str().unwrap().to_string(), blog_name.to_string()),
                                                    blog_account: blog_name.to_string(),
                                                    article_id: x["entry_id"].as_u64().unwrap(),
                                                };
                                                list.push(page_data)
                                            }
                                        }
                                    }
                                }
                                );
                            }
                            Err(err) => {
                                eprintln!("Failed to parse json string: {}", err);
                                async_wait(1500).await;
                            }
                        }
                    }
                    Err(err) => {
                        eprintln!("Failed to get json string: {}", err);
                        async_wait(1500).await;
                    }
                }
            }
            Err(err) => {
                eprintln!("Failed to get list page.: {}", err);
                async_wait(1500).await;
            }
        }
    }
    // drop(_permit);
    page_urls.unwrap()
}

fn theme_curator(theme: String, blog_id: String) -> String {
    let theme_val;
    match blog_id.as_str() {
        // "" => theme_val = "null".to_owned(),
        "risa-ogata" => theme_val = "小片リサ".to_owned(),
        "shimizu--saki" => theme_val = "清水佐紀".to_owned(),
        "kumai-yurina-blog" => theme_val = "熊井友理奈".to_owned(),
        "sudou-maasa-blog" => theme_val = "須藤茉麻".to_owned(),
        "sugaya-risako-blog" => theme_val = "菅谷梨沙子".to_owned(),
        "miyamotokarin-official" => theme_val = "宮本佳林".to_owned(),
        "sayumimichishige-blog" => theme_val = "道重さゆみ".to_owned(),
        "kudo--haruka" => theme_val = "工藤遥".to_owned(),
        "airisuzuki-officialblog" => theme_val = "鈴木愛理".to_owned(),
        "angerme-ayakawada" => theme_val = "和田彩花".to_owned(),
        "miyazaki-yuka-blog" => theme_val = "宮崎由加".to_owned(),
        "tsugunaga-momoko-blog" => theme_val = "嗣永桃子".to_owned(),
        "natsuyaki-miyabi-blog" => theme_val = "夏焼雅".to_owned(),
        "tokunaga-chinami-blog" => theme_val = "徳永千奈美".to_owned(),
        "tanakareina-blog" => theme_val = "田中れいな".to_owned(),
        "ozeki-mai-official" => theme_val = "小関舞".to_owned(),
        _ => theme_val = theme
    }
    if theme_val == "梁川 奈々美" {
        "梁川奈々美".to_owned()
    } else {
        theme_val
    }
}

async fn parse_article_page(client: Client, page_data: PageData, matcher: Regex, semaphore: Arc<Semaphore>) -> Vec<ImageData> {
    let mut page_urls: Option<Vec<ImageData>> = None;
    let _permit = semaphore.acquire().await.unwrap();
    // println!("start: {}", page_data.blog_page);
    while page_urls.is_none() {
        match client.get(page_data.blog_page.as_str())
            .send().await {
            Ok(resp) => {
                let resp_text = resp.text().await.unwrap().as_str().to_string();
                match matcher.captures(&*resp_text).unwrap().get(0).ok_or(&resp_text) {
                    Ok(matched_text) => {
                        let mut json_str = matched_text.as_str().to_string();
                        json_str.pop();
                        json_str = json_str.replacen("<script>window.INIT_DATA=", "", 1);
                        match serde_json::from_str::<Value>(&json_str) {
                            Ok(json) => {
                                let article_val = json["entryState"]["entryMap"].as_object().unwrap().values().cloned().collect::<Vec<_>>();
                                // let theme_name
                                let article_html = article_val.get(0).unwrap()["entry_text"].clone();
                                // let html = Html::parse_fragment(article_html.as_str().unwrap());
                                let content_text = html_to_text(article_html, article_val.get(0).unwrap().clone(), page_data.clone());
                                page_urls = Some(content_text.0.clone());
                                // println!("{:?}", content_text);
                                // exit(0);
                            }
                            Err(err) => {
                                eprintln!("Failed to parse json string: {}", err);
                                async_wait(1500).await;
                            }
                        }
                    }
                    Err(err) => {
                        eprintln!("Failed to get json string: {}", err);
                        async_wait(1500).await;
                    }
                }
            }
            Err(err) => {
                eprintln!("Failed to get list page.: {}", err);
                async_wait(1500).await;
            }
        }
    }
    // drop(permit);
    // println!("end: {}", page_data.blog_page);
    page_urls.unwrap().clone()
}

async fn download_file(client: Client, url: String, filename: String, semaphore: Arc<Semaphore>, modified_time: DateTime<FixedOffset>) {
    let _permit = semaphore.acquire().await.unwrap();

    match client.get(url).send().await {
        Ok(resp) => {
            let mut file = async_fs::File::create(filename.clone()).await.unwrap();
            file.write_all(resp.bytes().await.unwrap().as_ref()).await.unwrap();
            set_file_times(filename.as_str().to_string(), FileTime::now(), FileTime::from(SystemTime::from(modified_time))).unwrap();
            // let metadata = file.metadata().await.unwrap();
            // metadata.modified().unwrap();

            file.flush().await.unwrap();
        }
        Err(err) => {
            eprintln!("{}", err);
            panic!()
        }
    }
}

#[tokio::main]
async fn main() {
    // console_subscriber::init();
    let client = Client::new();
    let semaphore = Arc::new(Semaphore::new(50));
    let mut tasks = Vec::new();
    let page_count: Regex = Regex::new(r"<script>window.INIT_DATA=(.*?)};").unwrap();

    for name in NAMES {
        let task = spawn(get_page_count(
            client.clone(),
            name,
            page_count.clone(),
        ));
        tasks.push(task);
    }

    let list_page_count: Vec<_> = future::join_all(tasks)
        .await
        .iter()
        .map(|x| x.as_ref().unwrap().clone())
        .collect();
    for (order, &i) in (&list_page_count).iter().enumerate() {
        println!("{}: {}", NAMES[order], i);
    }
    let mut tasks = vec![];

    for i in 0..NAMES.len() {
        for j in 1..=list_page_count[i] {
            // println!("https://ameblo.jp/{}/entrylist-{}.html", NAMES[i], j);
            let task = spawn(
                parse_list_page(client.clone(), NAMES[i],
                                j, page_count.clone(), semaphore.clone())
            );
            tasks.push(task);
        }
    }
    let mut all_articles = vec![];
    let binding = future::join_all(tasks).await;
    for item in binding.iter().map(|x| { x.as_ref().unwrap() }).collect::<Vec<_>>() {
        all_articles.extend(item);
        // exit(-1);
    }
    println!("{}", all_articles.len());
    let mut tasks = vec![];
    // let semaphore = Arc::new(Semaphore::new(2));
    for url in all_articles {
        let task = spawn(
            parse_article_page(client.clone(), url.clone(), page_count.clone(), semaphore.clone())
        );
        tasks.push(task);

        // println!("{:?}", url);
    }
    let mut tasks_dl = vec![];

    if !Path::is_dir(Path::new(".").join("images").as_path()) {
        create_dir(Path::new(".").join("images").as_path()).unwrap();
    }
    future::join_all(tasks).await.iter().for_each(|x| {
        x.as_ref().unwrap().iter().for_each(|x| {
            // let filename = x.filename.clone();
            let url = x.url.clone();
            if !Path::is_dir(Path::new(".").join("images").join(&x.theme).as_path()) {
                create_dir(Path::new(".").join("images").join(&x.theme).as_path()).unwrap();
            }
            let file_path = Path::new(".").join("images").join(&x.theme).join(&x.filename);
            let task = spawn(
                download_file(client.clone(), url, file_path.to_str().unwrap().to_string(), semaphore.clone(), x.page_data.last_edit_datetime)
            );
            tasks_dl.push(task);
            println!("{} {} {} ", x.url, x.filename, x.date.with_timezone(&FixedOffset::east_opt(9 * 3600).unwrap()));
        })
    });
    future::join_all(tasks_dl).await;
}
