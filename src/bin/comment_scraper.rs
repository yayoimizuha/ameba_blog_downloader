use std::collections::HashMap;
use chrono::{DateTime, FixedOffset};
use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
struct CommentElement {
    comment_id: i64,
    blog_id: i64,
    entry_id: i64,
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

#[tokio::main]
async fn main() {
    let client = Client::new();
    let res = client.get("https://ameblo.jp/_api/blogComments;amebaId=juicejuice-official;blogId=10039630379;entryId=11618721121;excludeReplies=false;limit=100;offset=0").send().await.unwrap();
    let deserialized = serde_json::from_str::<CommentsJson>(res.text().await.unwrap().as_str()).unwrap();
    println!("{:?}", deserialized);
}