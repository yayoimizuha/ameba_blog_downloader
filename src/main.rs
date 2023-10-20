use std::ops::Deref;
use futures::future;
use futures::stream::iter;
use reqwest::Client;
use tokio::{task, time};
use regex::Regex;

const NAMES: &[&str] = &["angerme-ss-shin", "angerme-amerika", "angerme-new", "juicejuice-official",
    "tsubaki-factory", "morningmusume-9ki", "morningmusume-10ki", "mm-12ki", "morningm-13ki",
    "morningmusume15ki", "morningmusume16ki", "beyooooonds-rfro", "beyooooonds-chicatetsu",
    "beyooooonds", "ocha-norma", "countrygirls", "risa-ogata", "kumai-yurina-blog",
    "sudou-maasa-blog", "sugaya-risako-blog", "miyamotokarin-official", "kobushi-factory",
    "sayumimichishige-blog", "kudo--haruka", "airisuzuki-officialblog", "angerme-ayakawada",
    "miyazaki-yuka-blog", "tsugunaga-momoko-blog", "tokunaga-chinami-blog", "c-ute-official",
    "tanakareina-blog"];

// const NAMES: &[&str] = &["angerme-ss-shin", "angerme-amerika"];


async fn get_page_count(client: Client, name: &str, _order: usize, page_count: Regex) -> u64 {
    let mut page_nums: Option<u64> = None;

    let async_wait = |t: u64| async move {
        time::sleep(time::Duration::from_millis(t)).await
    };
    while page_nums.is_none() {
        match client.get(&format!("https://ameblo.jp/{name}/entrylist.html"))
            .send().await {
            Ok(response) => {
                let text = response.text().await.unwrap();
                match page_count.captures(&*text).unwrap().get(0).
                    ok_or(&text) {
                    Ok(matched_text) => {
                        let mut json_str = matched_text.as_str().to_string();
                        json_str.pop();
                        json_str = json_str.replacen("<script>window.INIT_DATA=", "", 1);
                        match serde_json::from_str::<serde_json::Value>(&json_str) {
                            Ok(json) => {
                                let _ = json["entryState"]["blogPageMap"].as_object().unwrap().iter()
                                    .map(|x| {
                                        page_nums = Option::from(x.1["paging"]["max_page"].as_u64().unwrap());
                                    }).collect::<Vec<_>>();
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


#[tokio::main]
async fn main() {
    let client = Client::new();
    let mut tasks = Vec::new();
    let page_count: Regex = Regex::new(r"<script>window.INIT_DATA=(.*?)};").unwrap();

    for i in 0..NAMES.len() {
        let task = task::spawn(get_page_count(client.clone(),
                                              NAMES[i], i, page_count.clone()));
        tasks.push(task);
    }

    let list_page_count: Vec<_> = future::join_all(tasks).await.iter()
        .map(|x| { x.as_ref().unwrap().clone() }).collect();
    for &i in &list_page_count {
        println!("{}", i);
    }
    for i in 0..NAMES.len() {
        for j in 1..list_page_count[i] {
            println!("https://ameblo.jp/{}/page-{}.html", NAMES[i], j);
        }
    }
}
