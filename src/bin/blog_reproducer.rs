use std::env::args;
use std::fmt::format;
use std::path::PathBuf;
use std::sync::Arc;
use chrono::DateTime;
use futures::future::join_all;
use itertools::Itertools;
use ort::tensor::Utf8Data;
use regex::{Captures, Regex};
use sqlx::SqlitePool;
use sqlx::sqlite::SqliteConnectOptions;
use tokio::fs::{create_dir, File};
use tokio::io::AsyncWriteExt;
use tokio::sync::Semaphore;
use ameba_blog_downloader::data_dir;

#[tokio::main]
async fn main() {
    let dest = match PathBuf::from(args().last().unwrap()).is_dir() {
        true => { PathBuf::from(args().last().unwrap()) }
        false => { unreachable!() }
    };
    let image_line_regex = Regex::new(r"-----image-----(\d*)-----").unwrap();
    let connection_option = SqliteConnectOptions::new().filename(data_dir().join("blog_text.sqlite"));
    let connection = SqlitePool::connect_with(connection_option).await.unwrap();
    let blog_keys = sqlx::query_as("SELECT DISTINCT blog_key FROM blog").fetch_all(&connection).await.unwrap().iter().map(|(v, ): &(String,)| { v.clone() }).collect::<Vec<_>>();
    println!("{:?}", blog_keys);
    let _ = join_all(blog_keys.iter().map(async |key| {
        if !dest.join(key).exists() {
            create_dir(dest.join(key)).await.unwrap();
        }
        if !dest.join(key).join("articles").exists() {
            create_dir(dest.join(key).join("articles")).await.unwrap();
        }
    }).collect::<Vec<_>>()).await;
    let semaphore = Arc::new(Semaphore::new(4));
    let _ = join_all(blog_keys.iter().map(async |key| {
        let themes = [
            vec!["全員".to_owned()],
            sqlx::query_as("SELECT DISTINCT theme FROM blog WHERE blog_key = ?;").bind(key).fetch_all(&connection).await.unwrap().iter().map(|(x, ): &(String,)| { x.clone() }).collect::<Vec<_>>()
        ].concat();
        join_all(themes.iter().map(async |theme| {
            let _permit = semaphore.acquire().await.unwrap();
            let (name, mut index_file) = match theme.as_str() {
                "全員" => { ("全員", File::create(dest.join(key).join("index.html")).await.unwrap()) }
                name => { (name, File::create(dest.join(key).join(name.to_owned() + ".html")).await.unwrap()) }
            };
            let navbar_html = themes.clone().iter().map(|theme| {
                if theme == name {
                    format!(r#"<a class="navbar">{theme}</a>"#)
                } else {
                    if theme == "全員" {
                        format!(r#"<a href="./index.html" class="navbar">{theme}</a>"#)
                    } else {
                        format!(r#"<a href="./{theme}.html" class="navbar">{theme}</a>"#)
                    }
                }
            }).join("\n");
            let table_body = match name == "全員" {
                true => { sqlx::query_as("SELECT DISTINCT article_id,theme,title,date FROM blog WHERE blog_key = ?;").bind(key) }
                false => { sqlx::query_as("SELECT DISTINCT article_id,theme,title,date FROM blog WHERE theme = ? AND blog_key = ?;").bind(name).bind(key) }
            }.fetch_all(&connection).await.unwrap().iter().map(|(id, theme, title, date): &(i64, String, String, String)| {
                (id.clone(), theme.clone(), title.clone(), DateTime::parse_from_rfc3339(date).unwrap())
            }).map(|(id, theme, title, date)| {
                format!(r#"<tr>
                <th scope="row">
                    <a href="./articles/{id}.html">{title}</a>
                </th>
                <td>
                    <a href="./{theme}.html">{theme}</a>
                </td>
                <td>
                    {0}
                </td>
            </tr>"#, date.format("%Y年%m月%d日　%H時%M分"))
            }).join("\n");
            index_file.write_all(format!(r#"<!DOCTYPE html>
<head>
    <link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Roboto:300,300italic,700,700italic">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/normalize/8.0.1/normalize.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/milligram/1.4.1/milligram.css">
    <style>
        .navbar {{
            font-size: max(3vw, 12px);
            padding: 10px;
        }}
    </style>
</head>

<body>
    <div>
        {navbar_html}
    </div>
        <table>
            <thead>
                <tr>
                    <th scope="col">タイトル</th>
                    <th scope="col">テーマ</th>
                    <th scope="col">日付</th>
                </tr>
            </thead>
        <tbody>
{table_body}
        </tbody>
    </table>

</body>"#).as_utf8_bytes()).await.unwrap();
        })).await;
        join_all(sqlx::query_as("SELECT DISTINCT article_id,theme,title,article FROM blog WHERE blog_key = ?;").bind(key).fetch_all(&connection).await.unwrap().iter().map(async |(id, theme, title, article): &(i64, String, String, String)| {
            File::create(dest.join(key).join("articles").join(format!("{id}.html"))).await.unwrap().write_all({
                let article = article.replace("\n", "<br>");
                let article = image_line_regex.replace_all(article.as_str(), |captures: &Captures| {
                    format!(r#"<image src="../../../blog_images/{theme}/{theme}={key}={id}-{}.jpg">"#, &captures[1].parse::<i32>().unwrap() - 1)
                }).to_string();
                format!(r#"<!DOCTYPE html>
<head>
    <link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Roboto:300,300italic,700,700italic">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/normalize/8.0.1/normalize.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/milligram/1.4.1/milligram.css">
    <style>
    </style>
</head>
<body>
    <h1>{title}</h1>
    {article}
</body>"#).as_utf8_bytes()
            }).await.unwrap();
        }).collect::<Vec<_>>()).await;
    }).collect::<Vec<_>>()).await;
}