use std::io::Read;
use futures::{StreamExt, stream, future};
use reqwest::{Client, Response};
use tokio::io::AsyncReadExt;
use tokio::task;

const NAMES: &[&str] = &["tsubaki-factory", "angerme-ss-shin"];

// implement get_text function
async fn get_text(client: Client, name: &str) -> String {
    // get response from the url
    let mut response = client.get(&format!("https://helloproject.fandom.com/wiki/{}", name))
        .send().await.unwrap();
    // read response body as text
    let mut text = String::new();
    response.read_to_string(&mut text).unwrap();
    // return text
    text
}

// main function

#[tokio::main]
async fn main() {
    let client = Client::new();
    let mut tasks = Vec::new();
    // get all NAMES contents concurrently using reqwest as text
    for name in NAMES {
        let task = task::spawn(get_text(client.clone(), name));
        tasks.push(task);
    }
    // get all tasks result concurrently
    let results = future::join_all(tasks).await;
    // print out all results
    for result in results {
        println!("{}", result.unwrap());
    }
}
