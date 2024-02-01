use crate::clients::{Document, EmbedRequest, EmbedResponse};
use anyhow::Result;
use curl::easy::{Easy, List, ReadError};
use curl::Error;
use serde::de::{EnumAccess, MapAccess, SeqAccess, Unexpected, Visitor};
use serde::{Deserialize, Deserializer};
use std::borrow::Cow;
use std::fmt::{Display, Formatter};
use std::io::{stdout, Read, Write};
use std::str::FromStr;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use reqwest::{Client, Response};
use tracing::field::debug;
use tracing::{debug, error, info, warn};

#[derive(PartialEq)]
pub enum Status {
    Ok,
    Loading,
    Error,
    Unknown,
}

impl Display for Status {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let string = match self {
            Status::Ok => "ok",
            Status::Loading => "loading",
            Status::Error => "error",
            Status::Unknown => "unknown",
        };

        f.write_str(string)
    }
}
impl<'de> Deserialize<'de> for Status {
    fn deserialize<D>(deserializer: D) -> Result<Status, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(StatusVisitor)
    }
}

struct StatusVisitor;
impl<'de> Visitor<'de> for StatusVisitor {
    type Value = Status;

    fn expecting(&self, formatter: &mut Formatter) -> std::fmt::Result {
        formatter.write_str("a slice around 16-32 bytes wide.")
    }

    fn visit_map<A>(self, mut map: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        while let Ok(Some((k, v))) = map.next_entry::<&str, &str>() {
            let stat = match v {
                "ok" => Status::Ok,
                "loading model" => Status::Loading,
                "error" => Status::Error,
                &_ => Status::Unknown,
            };

            return Ok(stat);
        }

        Ok(Status::Unknown)
    }
}

// Returns a slice of the array ending at the first zero byte found:
// Serde doesn't appreciate strings with dangling zeros.
fn fit_to_size(buf: &[u8]) -> &[u8] {
    for i in 0..buf.len() {
        if buf[i] == 0 {
            return &buf[0..i];
        }
    }

    &buf
}

pub struct LlamaCpp<'l> {
    pub(crate) https: bool,
    pub(crate) host: &'l str,
    pub(crate) port: u16,
    pub(crate) headers: List,
    client: Client
}

impl<'l> Default for LlamaCpp<'l> {
    fn default() -> Self {
        let headers = {
            let mut l = List::new();
            _ = l.append("Accept: application/json");
            _ = l.append("Content-Type: application/json");
            l
        };

        Self {
            https: false,
            host: "127.0.0.1",
            port: 8080,
            headers,
            client: Client::new()
        }
    }
}

impl<'l> LlamaCpp<'l> {
    pub fn new(host: &'l str, port: u16, headers: List, https: bool) -> Self {
        Self {
            https,
            host,
            port,
            headers,
            client: reqwest::Client::new(),
        }
    }

    fn create_url(&self, endpoint: &str) -> String {
        let mut http = String::from("http");
        if self.https {
            http.push('s');
        }

        format!("{}://{}:{}/{}", &http, &self.host, &self.port, endpoint)
    }

    fn clone_headers(&self) -> Result<List> {
        let mut l = List::new();

        for item in self.headers.iter() {
            let s = std::str::from_utf8(item)?;
            _ = l.append(s);
        }

        Ok(l)
    }



    pub fn health_check(&self) -> Result<Status> {
        info!("Performing health check");
        let buf: Arc<Mutex<[u8; 32]>> = Arc::new(Mutex::new([0u8; 32]));
        let buf_c: Arc<Mutex<[u8; 32]>> = buf.clone();
        let url = self.create_url("health");
        let mut curl = Easy::new();

        _ = curl.url(&url)?;
        _ = curl.http_headers(self.clone_headers()?);
        // curl.perform() blocks so Arc guards are just to make the compiler happy
        _ = curl.write_function(move |dataz| {
            let mut buf = buf_c.lock().unwrap();
            let l = dataz.len();

            if l > 32 {
                let s = std::str::from_utf8(dataz).unwrap_or("RT: Failed to decode body via utf-8");
                warn!("Oddly sized health message:");
                warn!("{}", s);
            } else {
                _ = buf[..l].copy_from_slice(dataz);
                debug!("Read {} bytes from remote", l);
            }

            Ok(l)
        })?;

        _ = curl.perform()?;

        let buf = buf.lock().unwrap();
        let slice = fit_to_size(&*buf);
        // let _strang = String::from_utf8_lossy(slice);
        let obj: Status = serde_json::from_slice(slice)?;

        info!("Llama is {obj}");

        return Ok(obj);
    }

    pub async fn embed(&self, text: Document) -> Result<Document> {
        // TODO POST via cURL - For some reason the write_function was losing bytes so reqwest it for now.
        let url = self.create_url("embedding");
        let request: EmbedRequest = text.to_owned().into();
        let req_str: String = serde_json::to_string(&request)?;
        let res = self.client.post(url).body(req_str).send().await?;
        let json_str = res.text().await?;
        // Qdrant demands f32 instead of 64.. curious
        let embedding_32 = match serde_json::from_str::<EmbedResponse>(&json_str) {
           Ok(response) => response.embedding
                .into_iter()
                .map(|f| f as f32)
                .collect::<Vec<f32>>(),
            Err(_) => vec![]
        };

        Ok(Document {
            page_content: text.page_content,
            metadata: text.metadata,
            embeddings: embedding_32,
        })
    }

    // TODO figure out why this thing's write_function loses bytes >:{
    // let mut curl = Easy::new();
    // _ = curl.url(&url);
    // _ = curl.post(true);
    // _ = curl.post_field_size(req_str.len() as u64)?;
    // _ = curl.http_headers(headers);
    // _ = curl.timeout(Duration::from_secs(10))?;
    //
    // let mut handle = curl.transfer();
    //
    // _ = handle.read_function(move |mut into| Ok(req_str.as_bytes().read(into).unwrap()))?;
    // _ = handle.write_function(move |bytes| {
    //     let mut buf = buf_c.lock().unwrap();
    //     let l = bytes.len();
    //
    //     if l > 65536 {
    //         let s = std::str::from_utf8(bytes).unwrap_or("RT: Failed to decode body via utf-8");
    //         warn!("Oversized message: {}", s);
    //     } else {
    //         _ = buf[0..l].copy_from_slice(bytes);
    //         info!("Read {} bytes from remote", l);
    //     }
    //
    //     Ok(l)
    // })?;
    // _ = handle.perform()?;
}
