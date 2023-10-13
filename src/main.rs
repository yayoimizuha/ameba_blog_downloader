use futures::future;
use reqwest::Client;
use tokio::task;
use regex::Regex;

// const NAMES: &[&str] = &["angerme-ss-shin", "angerme-amerika", "angerme-new", "juicejuice-official",
//     "tsubaki-factory", "morningmusume-9ki", "morningmusume-10ki", "mm-12ki", "morningm-13ki",
//     "morningmusume15ki", "morningmusume16ki", "beyooooonds-rfro", "beyooooonds-chicatetsu",
//     "beyooooonds", "ocha-norma", "countrygirls", "risa-ogata", "kumai-yurina-blog",
//     "sudou-maasa-blog", "sugaya-risako-blog", "miyamotokarin-official", "kobushi-factory",
//     "sayumimichishige-blog", "kudo--haruka", "airisuzuki-officialblog", "angerme-ayakawada",
//     "miyazaki-yuka-blog", "tsugunaga-momoko-blog", "tokunaga-chinami-blog", "c-ute-official",
//     "tanakareina-blog"];
const NAMES: &[&str] = &["angerme-ss-shin", "angerme-amerika"];


async fn get_page_count(client: Client, name: &str, order: usize, page_count: Regex) -> String {
    // println!("start: {order}_{name}");
    let response = client.get(&format!("https://ameblo.jp/{name}/entrylist.html"))
        .send().await.unwrap();
    let text = response.text().await.unwrap();
    println!("end: {order}_{name}");
    let mut json = page_count.captures(&text).unwrap().get(0).unwrap().as_str().to_string();
    json.pop();
    let _json = json.replacen("<script>window.INIT_DATA=", "", 1);
    let json_parse: serde_json::Value = serde_json::from_str(&_json).unwrap();
    println!("page count: {:?}", json_parse["entryState"]["blogPageMap"]);

    text
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
    let _results = future::join_all(tasks).await;
}
