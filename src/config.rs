use anyhow::Result;
use hf_hub::{Repo, RepoType, api::sync::Api};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractorConfig {
    pub model_name: String,
    pub max_width: usize,
    pub counting_layer: String,
    pub token_pooling: Option<String>,
    pub max_len: Option<usize>,
}

impl Default for ExtractorConfig {
    fn default() -> Self {
        Self {
            model_name: "microsoft/deberta-v3-base".to_string(),
            max_width: 8,
            counting_layer: "count_lstm".to_string(),
            token_pooling: None,
            max_len: None,
        }
    }
}

pub struct ModelFiles {
    pub config: PathBuf,
    pub encoder_config: PathBuf,
    pub tokenizer: PathBuf,
    pub weights: PathBuf,
}

pub fn download_model(repo_id: &str) -> Result<ModelFiles> {
    let api = Api::new()?;
    let repo = api.repo(Repo::new(repo_id.to_string(), RepoType::Model));

    let config = repo.get("config.json")?;
    let encoder_config = repo.get("encoder_config/config.json")?;
    let tokenizer = repo.get("tokenizer.json")?;
    let weights = repo.get("model.safetensors")?;

    Ok(ModelFiles {
        config,
        encoder_config,
        tokenizer,
        weights,
    })
}
