mod clients;

use std::collections::VecDeque;
use std::default::Default;
use std::path::PathBuf;
use std::thread::JoinHandle;
use std::time::Duration;

use anyhow::Result;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use indicatif::style::TemplateError;
use qdrant_client::client::{QdrantClient, QdrantClientConfig};
use qdrant_client::prelude::CreateCollection;
use qdrant_client::qdrant::{PointId, PointStruct, Vector, Vectors};
use qdrant_client::qdrant::point_id::PointIdOptions;
use qdrant_client::qdrant::vectors::VectorsOptions;
use tokio::fs::File;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::runtime::Runtime;
use tokio::sync::mpsc;
use tokio::sync::mpsc::{Receiver, UnboundedReceiver};
use tracing::{info, warn};
use uuid::Uuid;
use crate::clients::Document;
use crate::clients::llm::llama_cpp::{LlamaCpp, Status};
use crate::clients::vector_store::qdrant::Qlient;

#[tokio::main]
async fn main() -> Result<()> {
    init_observation();

    let (tx, mut rx) = mpsc::unbounded_channel::<Result<Document>>();
    let llama = LlamaCpp::default();
    let documents = read_documents(
        "/home/echo/projects/llms/documents".into()
    ).await?;

    info!("Read {} documents from storage", documents.len());

    await_llama(&llama).await;

    let qdrant_handle = vector_upsert_loop(documents.len() as u64, rx);

    for document in documents.into_iter() {
        let embedded= llama.embed(document).await;
         _ = tx.send(embedded);
    }

    drop(tx);

    _ = qdrant_handle.join();

    Ok(())
}

fn init_observation() {
    // TODO Flamegraph (tracing-flame)
    // TODO metrics output
    tracing_subscriber::fmt().init();
}

/// Waits for the Llama.cpp server to acknowledge a ready model
async fn await_llama(llama: &LlamaCpp<'_>) -> Result<()> {
    let mut dur = Duration::from_secs(7);

    while llama.health_check()? != Status::Ok {
        tokio::time::sleep(dur).await;
        dur += Duration::from_millis(500)
    }

    Ok(())
}

/// Instantiates the event loop for handing embedded documents to the Qdrant client
fn vector_upsert_loop(total_expected: u64, mut rx: UnboundedReceiver<Result<Document>>) -> JoinHandle<()> {
    std::thread::spawn(move || Runtime::new()
        .expect("Something is very wrong")
        .block_on(async move {
            let mut client = Qlient::default();
            if !client.client.has_collection("rust2").await.expect("oopsies daises") {
                let _ = client.client.create_collection(
                    &CreateCollection {
                        collection_name: "rust2".to_string(),
                        hnsw_config: None,
                        wal_config: None,
                        optimizers_config: None,
                        shard_number: None,
                        on_disk_payload: None,
                        timeout: None,
                        vectors_config: None,
                        replication_factor: None,
                        write_consistency_factor: None,
                        init_from_collection: None,
                        quantization_config: None,
                        sharding_method: None,
                        sparse_vectors_config: None,
                    }
                ).await;
            }
            let mut prog_bars = MultiProgress::new();

            let processed = prog_bars.add(progress_bar(
                total_expected.clone(),
                Some("{pos} processed".to_string())).unwrap());
            let errors = prog_bars.add(progress_bar(
                total_expected.clone(),
                Some("{pos} failures".to_string())).unwrap());
            let embeddings = prog_bars.add(progress_bar(
                total_expected.clone(),
                Some("{pos} embeddings generated".to_string())).unwrap());
            let stored = prog_bars.add(progress_bar(
                total_expected,
                Some("{pos} embeddings stored".to_string())).unwrap());

            while let Some(result) = rx.recv().await {
                match result {
                    Ok(document) if !document.embeddings.is_empty() => {
                        embeddings.inc(1);
                        match client.push(document).await {
                            Ok(_) => {
                                stored.inc(1);
                            }
                            Err(_) => {
                                errors.inc(1);
                            }
                        }
                    },
                    Ok(_) => {
                        processed.inc(1);
                    },
                    Err(_) => {
                        errors.inc(1);
                    }
                }
            }

            _  = prog_bars.clear();
        }))
}

/// Reads Documents from local storage into a VecDeque
async fn read_documents(path: PathBuf) -> Result<VecDeque<Document>> {
    let mut vec = VecDeque::new();

    let mut file = File::open(path).await?;
    let mut buffer = BufReader::new(file);
    let mut lines = buffer.lines();

    while let Ok(Some(k)) = lines.next_line().await {
        if let Ok(doc) = serde_json::from_str::<Document>(&k) {
            vec.push_front(doc);
        }
    }

    Ok(vec)
}


/// Creates an indicatif prog bar via `style_template`
fn progress_bar(len: u64, style_template: Option<String>) -> Result<ProgressBar> {
    let template = style_template
        .unwrap_or("ETA: {eta_precise}\nElapsed: {elapsed_precise}\n{per_sec} {wide_bar} {pos}/{len}".to_string());
    let style = ProgressStyle::with_template(&template)?;

    Ok(ProgressBar::new(len).with_style(style))
}

