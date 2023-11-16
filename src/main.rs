use std::fs::File;
use std::io::Write;
use std::process::exit;
use std::str::FromStr;
use std::sync::Arc;
use chrono::{DateTime, Utc};
use futures::future;
use html5ever::tree_builder::TreeSink;
use regex::Regex;
use reqwest::Client;
use serde_json::Value;
use tokio::{task, time};
use tokio::sync::Semaphore;
use scraper::{Html, Selector};

// const NAMES: &[&str] = &["angerme-ss-shin", "angerme-amerika", "angerme-new", "juicejuice-official",
//     "tsubaki-factory", "morningmusume-9ki", "morningmusume-10ki", "mm-12ki", "morningm-13ki",
//     "morningmusume15ki", "morningmusume16ki", "beyooooonds-rfro", "beyooooonds-chicatetsu",
//     "beyooooonds", "ocha-norma", "countrygirls", "risa-ogata", "kumai-yurina-blog",
//     "sudou-maasa-blog", "sugaya-risako-blog", "miyamotokarin-official", "kobushi-factory",
//     "sayumimichishige-blog", "kudo--haruka", "airisuzuki-officialblog", "angerme-ayakawada",
//     "miyazaki-yuka-blog", "tsugunaga-momoko-blog", "tokunaga-chinami-blog", "c-ute-official",
//     "tanakareina-blog", ];

const NAMES: &[&str] = &["airisuzuki-officialblog"];

async fn async_wait(t: u64) { time::sleep(time::Duration::from_millis(t)).await }

#[derive(Clone, Debug)]
struct PageData {
    blog_page: String,
    comment_api: String,
}

#[derive(Clone, Debug)]
struct ImageData {
    page_data: PageData,
    filename: String,
    url: String,
    date: chrono::DateTime<Utc>,
}


fn html_to_text(_html: Value, json: Value, page_data: PageData) -> Vec<String> {
    let mut html = Html::parse_document(_html.as_str().unwrap());
    let last_edit_date = DateTime::<Utc>::from_str(json["last_edit_datetime"].as_str().unwrap()).unwrap();
    // println!("{:?}", html.html());
    let image: Selector = Selector::parse("img[class=PhotoSwipeImage]").unwrap();
    // let emoji: Selector = Selector::parse("img.PhotoSwipeImage[data-src]").unwrap();
    // let texts = html.select(&all_text).next().unwrap().text().collect::<Vec<_>>();
    let emoji_selector = Selector::parse("img.emoji").unwrap();
    let emojis = html.select(&emoji_selector).map(|x|x.id()).collect::<Vec<_>>();
    for emoji in emojis {
        html.remove_from_parent(&emoji);
    }
    let mut tmp = html.select(&image);
    // println!("{:?}", tmp.map(|x| x.html()).collect::<Vec<_>>());
    // println!("{:?}", tmp);
    tmp.map(|x| {
        // println!("{}", x.html().as_str());
        x.html()
    }).collect::<Vec<_>>()
    // tmp.next().unwrap().html()
    // texts.join("\n")
    // "".to_string()
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
                        match serde_json::from_str::<serde_json::Value>(&json_str) {
                            Ok(json) => {
                                let _ = json["entryState"]["blogPageMap"]
                                    .as_object()
                                    .unwrap()
                                    .iter()
                                    .map(|x| {
                                        page_nums = Option::from(
                                            x.1["paging"]["max_page"].as_u64().unwrap(),
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
    let permit = semaphore.acquire().await.unwrap();
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
    drop(permit);
    page_urls.unwrap()
}


async fn parse_article_page(client: Client, page_data: PageData, matcher: Regex, semaphore: Arc<Semaphore>) -> Vec<String> {
    let mut page_urls: Option<Vec<String>> = None;
    let permit = semaphore.acquire().await.unwrap();
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
                                let article_html = article_val.get(0).unwrap()["entry_text"].clone();
                                let mut html_to_file = File::create("test.html").unwrap();
                                let mut text_to_file = File::create("test.txt").unwrap();
                                html_to_file.write_all(article_html.as_str().unwrap().as_bytes()).unwrap();
                                html_to_file.flush().unwrap();
                                // let html = Html::parse_fragment(article_html.as_str().unwrap());
                                let content_text = html_to_text(article_html, article_val.get(0).unwrap().clone(), page_data.clone());
                                text_to_file.write_all(content_text.join("\n").as_ref()).unwrap();
                                text_to_file.flush().unwrap();
                                page_urls = Some(content_text.clone());
                                println!("{:?}", content_text);
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
    drop(permit);
    page_urls.unwrap().clone()
}

#[tokio::main]
async fn main() {
    // console_subscriber::init();
    let client = Client::new();
    let semaphore = Arc::new(Semaphore::new(100));
    let mut tasks = Vec::new();
    let page_count: Regex = Regex::new(r"<script>window.INIT_DATA=(.*?)};").unwrap();

    for name in NAMES {
        let task = task::spawn(get_page_count(
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
    for &i in &list_page_count {
        println!("{}", i);
    }
    let mut tasks = vec![];
    for i in 0..NAMES.len() {
        for j in 1..=list_page_count[i] {
            // println!("https://ameblo.jp/{}/entrylist-{}.html", NAMES[i], j);
            let task = tokio::spawn(
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
    for url in all_articles {
        let task = tokio::spawn(
            parse_article_page(client.clone(), url.clone(), page_count.clone(), semaphore.clone())
        );
        tasks.push(task);

        println!("{:?}", url);
    }
    future::join_all(tasks).await;
}
