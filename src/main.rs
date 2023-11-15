use std::process::exit;
use std::sync::Arc;
use futures::future;
use regex::Regex;
use reqwest::Client;
use serde_json::Value;
use tokio::{task, time};
use tokio::sync::Semaphore;

// const NAMES: &[&str] = &["angerme-ss-shin", "angerme-amerika", "angerme-new", "juicejuice-official",
//     "tsubaki-factory", "morningmusume-9ki", "morningmusume-10ki", "mm-12ki", "morningm-13ki",
//     "morningmusume15ki", "morningmusume16ki", "beyooooonds-rfro", "beyooooonds-chicatetsu",
//     "beyooooonds", "ocha-norma", "countrygirls", "risa-ogata", "kumai-yurina-blog",
//     "sudou-maasa-blog", "sugaya-risako-blog", "miyamotokarin-official", "kobushi-factory",
//     "sayumimichishige-blog", "kudo--haruka", "airisuzuki-officialblog", "angerme-ayakawada",
//     "miyazaki-yuka-blog", "tsugunaga-momoko-blog", "tokunaga-chinami-blog", "c-ute-official",
//     "tanakareina-blog", ];


async fn async_wait(t: u64) { time::sleep(time::Duration::from_millis(t)).await }

struct PageData {
    blog_page: String,
    comment_api: String,
}

const NAMES: &[&str] = &["miyazaki-yuka-blog"];

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

async fn parse_list_page(client: Client, blog_name: &str, page: u64, matcher: Regex, semaphore: Arc<Semaphore>) -> Vec<String> {
    let mut page_urls: Option<Vec<String>> = None;
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
                                                list.push(article_url)
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
    page_urls.unwrap().clone()
}


async fn parse_article_page(client: Client, page_url: String, matcher: Regex, semaphore: Arc<Semaphore>) -> Vec<String> {
    let mut page_urls: Option<Vec<String>> = None;
    let permit = semaphore.acquire().await.unwrap();
    while page_urls.is_none() {
        match client.get(page_url.as_str())
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
                                println!("{}", article_val.get(0).unwrap()["entry_text"]);
                                exit(-1);
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
    let semaphore = Arc::new(Semaphore::new(300));
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
    for item in future::join_all(tasks).await.iter().map(|x| { x.as_ref().unwrap().clone() }).collect::<Vec<_>>() {
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

        println!("{}", url);
    }
    future::join_all(tasks).await;
}
