use std::collections::VecDeque;
use std::default::Default;
use std::fmt::{Debug, Formatter};
use anyhow::Result;
use qdrant_client::prelude::*;
use qdrant_client::prelude::point_id::PointIdOptions;
use qdrant_client::qdrant::{PointId, PointsOperationResponse, Vector, Vectors, WriteOrdering};
use qdrant_client::qdrant::shard_key::Key;
use qdrant_client::qdrant::vectors::VectorsOptions;
use tracing::{info, warn};
use uuid::Uuid;
use crate::clients::Document;

pub const DEFAULT_URI: &str = "http://localhost:6334";
pub const DEFAULT_BUFFER_SIZE: usize = 128;

pub struct Qlient {
    buffer: VecDeque<PointStruct>,
    size: usize,
    pub client: QdrantClient,
    collection_name: String,
    shard_key_selector: Option<Vec<Key>>,
    ordering: Option<WriteOrdering>,
}

impl Default for Qlient {
    fn default() -> Self {
        let buffer = VecDeque::with_capacity(DEFAULT_BUFFER_SIZE);
        let config = QdrantClientConfig::from_url(DEFAULT_URI);
        let client = QdrantClient::new(Some(config)).expect("failure will robinson!");

        Self {
            buffer,
            size: DEFAULT_BUFFER_SIZE,
            client,
            collection_name: "rust2".to_string(),
            shard_key_selector: None,
            ordering: None
        }
    }
}

impl Debug for Qlient {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.client.cfg.uri)
    }
}

impl Qlient {
    pub fn new(
        size: usize,
        config: Option<QdrantClientConfig>,
        collection_name: impl ToString,
        shard_key_selector: Option<Vec<Key>>,
        ordering: Option<WriteOrdering>,
    ) -> Self {
        let collection_name = collection_name.to_string();
        let buffer = VecDeque::with_capacity(size.clone());
        let client = QdrantClient::new(config)
            .expect("failure will robinson!");

        Self { buffer, size, client, collection_name, shard_key_selector, ordering }
    }

    pub async fn push(&mut self, document: Document) -> Result<()> {
        if self.buffer.len() < self.size {
            let uuid = Uuid::new_v4().to_string();
            let p_struct = document_to_pointstruct(uuid, document);
            self.buffer.push_front(p_struct);
            return Ok(())
        }

        if self.buffer.len() >= self.size {
            let points = self.buffer.drain(0..).collect();
            let result = self.client.upsert_points(
                self.collection_name.clone(),
                self.shard_key_selector.clone(),
                points,
                self.ordering.clone()
            ).await;

            match result {
                Ok(response) => {
                    if let Some(result) = response.result {
                        match result.status {
                            1 => Ok(()),
                            _ => Err(()),
                        }
                    }

                    Ok(())
                }
                Err(e) => {
                    warn!("{:?}", e);
                    Err(e)
                },
            }
        } else {
            Ok(())
        }
    }
}

#[inline]
fn document_to_pointstruct(uuid: impl ToString, d: Document) -> PointStruct {
    PointStruct {
        id: Some(PointId {
            point_id_options: Some(PointIdOptions::Uuid(uuid.to_string()))
        }),
        payload: d.metadata.into(),
        vectors: Some(Vectors {
            vectors_options: Some(VectorsOptions::Vector(Vector {
                data: d.embeddings,
                indices: None
            }))
        })
    }
}