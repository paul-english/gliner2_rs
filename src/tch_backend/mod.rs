//! tch-rs + rust-bert DeBERTa encoder with Candle heads (shared [`crate::model::Extractor`] weights for span/count/classifier).
//!
//! Encoder forward runs in LibTorch via [`rust_bert::deberta_v2::DebertaV2Model`]; activations are copied to Candle tensors for the existing head pipeline, giving numerical parity with the pure-Candle path without reimplementing GLiNER heads in tch.

mod loader;

use crate::config::{ExtractorConfig, ModelFiles};
use crate::engine::Gliner2Engine;
use crate::extract::{BatchSchemaMode, ExtractOptions};
use crate::model::Extractor;
use crate::processor::{FormattedInput, SchemaTransformer};
use crate::schema::Schema;
use anyhow::{Context, Result};
use candle_core::{Result as CResult, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::debertav2::Config as CandleDebertaConfig;
use rust_bert::deberta_v2::{DebertaV2Config, DebertaV2Model};
use serde_json::{Value, json};
use std::path::Path;
use tch::{Device as TchDevice, nn};

fn candle_err(e: candle_core::Error) -> anyhow::Error {
    anyhow::anyhow!("{e}")
}

/// GLiNER2 inference with a LibTorch DeBERTa encoder and Candle span/count heads.
pub struct TchExtractor {
    /// Keeps LibTorch encoder weights alive for `deberta`.
    #[allow(dead_code)]
    vs: nn::VarStore,
    deberta: DebertaV2Model,
    candle: Extractor,
    device_tch: TchDevice,
}

impl TchExtractor {
    /// Load from the same file layout as [`Extractor::load`] (mmap Candle heads + tch encoder from `model.safetensors`).
    pub fn load(
        files: &ModelFiles,
        extract_config: ExtractorConfig,
        mut encoder_config: CandleDebertaConfig,
        processor_vocab_size: usize,
        device_tch: TchDevice,
    ) -> Result<Self> {
        let dtype = candle_core::DType::F32;
        let cdev = candle_core::Device::Cpu;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[files.weights.clone()], dtype, &cdev)
                .map_err(|e| anyhow::anyhow!(e.to_string()))?
        };
        encoder_config.vocab_size = processor_vocab_size;
        let candle = Extractor::load(extract_config, encoder_config, vb)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;

        let enc_json = std::fs::read_to_string(&files.encoder_config)
            .with_context(|| format!("read {}", files.encoder_config.display()))?;
        let rb_cfg: DebertaV2Config =
            serde_json::from_str(&enc_json).context("parse encoder_config as DebertaV2Config")?;

        let mut vs = nn::VarStore::new(device_tch);
        let deberta = DebertaV2Model::new(&vs.root().sub("encoder"), &rb_cfg);
        vs.load(&files.weights)
            .map_err(|e| anyhow::anyhow!("tch VarStore::load: {e}"))?;

        Ok(Self {
            vs,
            deberta,
            candle,
            device_tch,
        })
    }

    /// Convenience: CPU LibTorch device.
    pub fn load_cpu(
        files: &ModelFiles,
        extract_config: ExtractorConfig,
        encoder_config: CandleDebertaConfig,
        processor_vocab_size: usize,
    ) -> Result<Self> {
        Self::load(
            files,
            extract_config,
            encoder_config,
            processor_vocab_size,
            TchDevice::Cpu,
        )
    }

    /// Load using `encoder_config.json` path (for tests / callers without [`ModelFiles`]).
    pub fn load_from_paths(
        weights: &Path,
        encoder_config_path: &Path,
        extract_config: ExtractorConfig,
        mut encoder_config: CandleDebertaConfig,
        processor_vocab_size: usize,
        device_tch: TchDevice,
    ) -> Result<Self> {
        let dtype = candle_core::DType::F32;
        let cdev = candle_core::Device::Cpu;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights.to_path_buf()], dtype, &cdev)
                .map_err(|e| anyhow::anyhow!(e.to_string()))?
        };
        encoder_config.vocab_size = processor_vocab_size;
        let candle = Extractor::load(extract_config, encoder_config, vb)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;

        let enc_json = std::fs::read_to_string(encoder_config_path)
            .with_context(|| format!("read {}", encoder_config_path.display()))?;
        let rb_cfg: DebertaV2Config =
            serde_json::from_str(&enc_json).context("parse encoder_config")?;

        let mut vs = nn::VarStore::new(device_tch);
        let deberta = DebertaV2Model::new(&vs.root().sub("encoder"), &rb_cfg);
        vs.load(weights)
            .map_err(|e| anyhow::anyhow!("tch VarStore::load: {e}"))?;

        Ok(Self {
            vs,
            deberta,
            candle,
            device_tch,
        })
    }

    /// Same NER PoC path as [`Extractor::forward`], but encoder runs in LibTorch.
    pub fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        formatted: &FormattedInput,
    ) -> CResult<Tensor> {
        let enc = self
            .encode_tch_to_candle(input_ids, attention_mask)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        self.candle.forward_from_encoder_output(&enc, formatted)
    }

    /// [`crate::batch_extract_entities`] equivalent for the tch encoder backend.
    pub fn batch_extract_entities(
        &self,
        transformer: &SchemaTransformer,
        texts: &[String],
        entity_types: &[String],
        opts: &ExtractOptions,
    ) -> Result<Vec<Value>> {
        let mut s = Schema::new();
        let types: Vec<Value> = entity_types.iter().map(|t| json!(t)).collect();
        s.entities(Value::Array(types));
        let (schema_val, meta) = s.build();
        crate::batch_extract(
            self,
            transformer,
            texts,
            BatchSchemaMode::Shared {
                schema: &schema_val,
                meta: &meta,
            },
            opts,
        )
    }

    fn encode_tch_to_candle(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let ids_tch = loader::candle_2d_int_to_tch(input_ids, self.device_tch)?;
        let mask_tch = loader::candle_2d_int_to_tch(attention_mask, self.device_tch)?;
        let token_type = tch::Tensor::zeros_like(&ids_tch);
        let out = tch::no_grad(|| {
            self.deberta.forward_t(
                Some(&ids_tch),
                Some(&mask_tch),
                Some(&token_type),
                None,
                None,
                false,
            )
        })
        .map_err(|e| anyhow::anyhow!("{e}"))?;
        loader::tch_hidden_to_candle(&out.hidden_state)
    }
}

impl Gliner2Engine for TchExtractor {
    type Tensor = Tensor;

    fn hidden_size(&self) -> usize {
        self.candle.hidden_size()
    }

    fn max_width(&self) -> usize {
        self.candle.max_width()
    }

    fn encode_sequence(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        self.encode_tch_to_candle(input_ids, attention_mask)
    }

    fn gather_text_word_embeddings(
        &self,
        last_hidden: &Tensor,
        positions: &[usize],
    ) -> Result<Tensor> {
        self.candle
            .gather_text_word_embeddings(last_hidden, positions)
            .map_err(candle_err)
    }

    fn gather_text_word_embeddings_batch_idx(
        &self,
        last_hidden: &Tensor,
        batch_idx: usize,
        positions: &[usize],
    ) -> Result<Tensor> {
        self.candle
            .gather_text_word_embeddings_batch_idx(last_hidden, batch_idx, positions)
            .map_err(candle_err)
    }

    fn compute_span_rep(&self, text_word_embs: &Tensor) -> Result<Tensor> {
        self.candle
            .compute_span_rep(text_word_embs)
            .map_err(candle_err)
    }

    fn compute_span_rep_batched(&self, token_embs_list: &[Tensor]) -> Result<Vec<Tensor>> {
        self.candle
            .compute_span_rep_batched(token_embs_list)
            .map_err(candle_err)
    }

    fn classifier_logits(&self, label_rows: &Tensor) -> Result<Tensor> {
        self.candle
            .classifier_logits(label_rows)
            .map_err(candle_err)
    }

    fn count_predict(&self, p_embedding: &Tensor) -> Result<usize> {
        self.candle.count_predict(p_embedding).map_err(candle_err)
    }

    fn span_scores_sigmoid(
        &self,
        span_rep: &Tensor,
        field_embs: &Tensor,
        pred_count: usize,
    ) -> Result<Tensor> {
        self.candle
            .span_scores_sigmoid(span_rep, field_embs, pred_count)
            .map_err(candle_err)
    }

    fn single_sample_inputs(&self, input_ids: &[u32]) -> Result<(Tensor, Tensor)> {
        Gliner2Engine::single_sample_inputs(&self.candle, input_ids)
    }

    fn batch_inputs(
        &self,
        input_ids: Vec<u32>,
        attention_mask_i64: Vec<i64>,
        batch_size: usize,
        max_seq_len: usize,
    ) -> Result<(Tensor, Tensor)> {
        Gliner2Engine::batch_inputs(
            &self.candle,
            input_ids,
            attention_mask_i64,
            batch_size,
            max_seq_len,
        )
    }

    fn batch_row_hidden(&self, hidden: &Tensor, idx: usize) -> Result<Tensor> {
        Gliner2Engine::batch_row_hidden(&self.candle, hidden, idx)
    }

    fn stack_schema_token_embeddings(
        &self,
        last_hidden_seq: &Tensor,
        positions: &[usize],
    ) -> Result<Tensor> {
        Gliner2Engine::stack_schema_token_embeddings(&self.candle, last_hidden_seq, positions)
    }

    fn tensor_dim0(&self, t: &Tensor) -> Result<usize> {
        Gliner2Engine::tensor_dim0(&self.candle, t)
    }

    fn tensor_narrow0(&self, t: &Tensor, start: usize, len: usize) -> Result<Tensor> {
        Gliner2Engine::tensor_narrow0(&self.candle, t, start, len)
    }

    fn tensor_index0(&self, t: &Tensor, i: usize) -> Result<Tensor> {
        Gliner2Engine::tensor_index0(&self.candle, t, i)
    }

    fn tensor_logits_1d(&self, logits: &Tensor) -> Result<Vec<f32>> {
        Gliner2Engine::tensor_logits_1d(&self.candle, logits)
    }

    fn tensor_span_scores_to_vec4(&self, t: &Tensor) -> Result<Vec<Vec<Vec<Vec<f32>>>>> {
        Gliner2Engine::tensor_span_scores_to_vec4(&self.candle, t)
    }
}
