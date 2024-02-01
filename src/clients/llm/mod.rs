pub mod llama_cpp;


pub trait Embedding {
    async fn embed_documents(text: &str);
    async fn embed_query(text: &str);
}