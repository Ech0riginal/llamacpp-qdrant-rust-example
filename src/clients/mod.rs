pub mod llm;
pub mod vector_store;

use std::collections::HashMap;
use qdrant_client::qdrant::Value;
use serde::{Serialize, Deserialize};

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Document {
    #[serde(rename = "page_content")]
    pub page_content: String,
    pub metadata: Metadata,
    #[serde(default)]
    pub embeddings: Vec<f32>,
}

impl Into<HashMap<String, Value>> for Metadata {
    fn into(self) -> HashMap<String, Value> {
        let mut map = HashMap::with_capacity(3);

        map.insert("source".to_string(), Value::from( self.source));
        map.insert("content_type".to_string(), Value::from( self.content_type));
        map.insert("language".to_string(), Value::from( self.language));

        map
    }
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Metadata {
    pub source: String,
    #[serde(rename = "content_type")]
    pub content_type: String,
    pub language: String,
}

impl Into<EmbedRequest> for Document {
    fn into(self) -> EmbedRequest {
        EmbedRequest {
            content: self.page_content
        }
    }
}

#[derive(Serialize)]
pub struct EmbedRequest {
    content: String
}

#[derive(Deserialize)]
pub struct EmbedResponse {
    pub embedding: Vec<f64>,
}