//! LibTorch DeBERTa encoder (`rust-bert`) + GLiNER heads on `tch::Tensor` (no Candle).
//!
//! Weights load from the same `model.safetensors` as the Candle backend: encoder keys into
//! `nn::VarStore`, head tensors via [`weights::load_safetensors`].

mod heads;
mod weights;

use crate::config::{ExtractorConfig, ModelFiles};
use crate::engine::Gliner2Engine;
use crate::processor::FormattedInput;
use anyhow::{Context, Result};
use heads::TchHeads;
use rust_bert::deberta_v2::{DebertaV2Config, DebertaV2Model};
use std::path::Path;
use tch::{Device as TchDevice, Kind, Tensor, nn};

pub struct TchExtractor {
    #[allow(dead_code)]
    vs: nn::VarStore,
    deberta: DebertaV2Model,
    heads: TchHeads,
    device_tch: TchDevice,
}

impl TchExtractor {
    pub fn load(
        files: &ModelFiles,
        extract_config: ExtractorConfig,
        processor_vocab_size: usize,
        device_tch: TchDevice,
    ) -> Result<Self> {
        let tm = weights::load_safetensors(&files.weights, device_tch)?;
        let heads = TchHeads::load(&tm.tensors, device_tch, &extract_config)?;

        let enc_json = std::fs::read_to_string(&files.encoder_config)
            .with_context(|| format!("read {}", files.encoder_config.display()))?;
        let mut rb_cfg: DebertaV2Config =
            serde_json::from_str(&enc_json).context("parse encoder_config as DebertaV2Config")?;
        rb_cfg.vocab_size = processor_vocab_size as i64;

        let mut vs = nn::VarStore::new(device_tch);
        let deberta = DebertaV2Model::new(vs.root().sub("encoder"), &rb_cfg);
        vs.load(&files.weights)
            .map_err(|e| anyhow::anyhow!("tch VarStore::load: {e}"))?;

        Ok(Self {
            vs,
            deberta,
            heads,
            device_tch,
        })
    }

    pub fn load_cpu(
        files: &ModelFiles,
        extract_config: ExtractorConfig,
        processor_vocab_size: usize,
    ) -> Result<Self> {
        Self::load(files, extract_config, processor_vocab_size, TchDevice::Cpu)
    }

    pub fn load_from_paths(
        weights: &Path,
        encoder_config_path: &Path,
        extract_config: ExtractorConfig,
        processor_vocab_size: usize,
        device_tch: TchDevice,
    ) -> Result<Self> {
        let tm = weights::load_safetensors(weights, device_tch)?;
        let heads = TchHeads::load(&tm.tensors, device_tch, &extract_config)?;

        let enc_json = std::fs::read_to_string(encoder_config_path)
            .with_context(|| format!("read {}", encoder_config_path.display()))?;
        let mut rb_cfg: DebertaV2Config =
            serde_json::from_str(&enc_json).context("parse encoder_config")?;
        rb_cfg.vocab_size = processor_vocab_size as i64;

        let mut vs = nn::VarStore::new(device_tch);
        let deberta = DebertaV2Model::new(vs.root().sub("encoder"), &rb_cfg);
        vs.load(weights)
            .map_err(|e| anyhow::anyhow!("tch VarStore::load: {e}"))?;

        Ok(Self {
            vs,
            deberta,
            heads,
            device_tch,
        })
    }

    /// NER-style forward: encoder + heads → `[num_entities, L, max_width]` scores.
    pub fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        formatted: &FormattedInput,
    ) -> Result<Tensor> {
        let enc = tch::no_grad(|| self.encode_sequence_internal(input_ids, attention_mask))?;
        Ok(self.heads.forward_from_encoder_output(
            &enc,
            &formatted.text_word_first_positions,
            &formatted.schema_special_positions,
        ))
    }

    fn encode_sequence_internal(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<Tensor> {
        let token_type = Tensor::zeros_like(input_ids);
        let out = self
            .deberta
            .forward_t(
                Some(input_ids),
                Some(attention_mask),
                Some(&token_type),
                None,
                None,
                false,
            )
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(out.hidden_state)
    }
}

impl Gliner2Engine for TchExtractor {
    type Tensor = Tensor;

    fn dup_tensor(&self, t: &Self::Tensor) -> Self::Tensor {
        t.shallow_clone()
    }

    fn hidden_size(&self) -> usize {
        self.heads.hidden_size
    }

    fn max_width(&self) -> usize {
        self.heads.max_width
    }

    fn encode_sequence(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        tch::no_grad(|| self.encode_sequence_internal(input_ids, attention_mask))
    }

    fn gather_text_word_embeddings(
        &self,
        last_hidden: &Tensor,
        positions: &[usize],
    ) -> Result<Tensor> {
        let row = last_hidden.select(0, 0);
        Ok(gather_row_positions(&row, positions))
    }

    fn gather_text_word_embeddings_batch_idx(
        &self,
        last_hidden: &Tensor,
        batch_idx: usize,
        positions: &[usize],
    ) -> Result<Tensor> {
        let row = last_hidden.select(0, batch_idx as i64);
        Ok(gather_row_positions(&row, positions))
    }

    fn compute_span_rep(&self, text_word_embs: &Tensor) -> Result<Tensor> {
        Ok(self.heads.compute_span_rep(text_word_embs))
    }

    fn compute_span_rep_batched(&self, token_embs_list: &[Tensor]) -> Result<Vec<Tensor>> {
        Ok(self.heads.compute_span_rep_batched(token_embs_list))
    }

    fn classifier_logits(&self, label_rows: &Tensor) -> Result<Tensor> {
        Ok(self.heads.classifier_logits(label_rows))
    }

    fn count_predict(&self, p_embedding: &Tensor) -> Result<usize> {
        Ok(self.heads.count_predict(p_embedding))
    }

    fn span_scores_sigmoid(
        &self,
        span_rep: &Tensor,
        field_embs: &Tensor,
        pred_count: usize,
    ) -> Result<Tensor> {
        Ok(self
            .heads
            .span_scores_sigmoid(span_rep, field_embs, pred_count))
    }

    fn single_sample_inputs(&self, input_ids: &[u32]) -> Result<(Tensor, Tensor)> {
        let dev = self.device_tch;
        let v: Vec<i64> = input_ids.iter().map(|&x| x as i64).collect();
        let t = Tensor::from_slice(&v).to_device(dev).unsqueeze(0);
        let mask = Tensor::ones(t.size().as_slice(), (Kind::Int64, dev));
        Ok((t, mask))
    }

    fn batch_inputs(
        &self,
        input_ids: Vec<u32>,
        attention_mask_i64: Vec<i64>,
        batch_size: usize,
        max_seq_len: usize,
    ) -> Result<(Tensor, Tensor)> {
        let dev = self.device_tch;
        let v: Vec<i64> = input_ids.into_iter().map(|x| x as i64).collect();
        let ids = Tensor::from_slice(&v)
            .to_device(dev)
            .reshape([batch_size as i64, max_seq_len as i64]);
        let mask = Tensor::from_slice(&attention_mask_i64)
            .to_device(dev)
            .reshape([batch_size as i64, max_seq_len as i64]);
        Ok((ids, mask))
    }

    fn batch_row_hidden(&self, hidden: &Tensor, idx: usize) -> Result<Tensor> {
        Ok(hidden.select(0, idx as i64))
    }

    fn stack_schema_token_embeddings(
        &self,
        last_hidden_seq: &Tensor,
        positions: &[usize],
    ) -> Result<Tensor> {
        let mut rows = Vec::new();
        for &p in positions {
            rows.push(last_hidden_seq.select(0, p as i64).unsqueeze(0));
        }
        Ok(Tensor::cat(&rows, 0))
    }

    fn tensor_dim0(&self, t: &Tensor) -> Result<usize> {
        Ok(t.size()[0] as usize)
    }

    fn tensor_narrow0(&self, t: &Tensor, start: usize, len: usize) -> Result<Tensor> {
        Ok(t.narrow(0, start as i64, len as i64))
    }

    fn tensor_index0(&self, t: &Tensor, i: usize) -> Result<Tensor> {
        Ok(t.select(0, i as i64))
    }

    fn tensor_logits_1d(&self, logits: &Tensor) -> Result<Vec<f32>> {
        let n = logits.numel();
        let mut v = vec![0f32; n];
        logits.copy_data(&mut v, n);
        Ok(v)
    }

    fn tensor_span_scores_to_vec4(&self, t: &Tensor) -> Result<Vec<Vec<Vec<Vec<f32>>>>> {
        let sz = t.size();
        if sz.len() != 4 {
            anyhow::bail!("expected 4D tensor");
        }
        let b = sz[0] as usize;
        let p = sz[1] as usize;
        let l = sz[2] as usize;
        let k = sz[3] as usize;
        let n = b * p * l * k;
        let mut flat = vec![0f32; n];
        t.copy_data(&mut flat, n);
        let mut out = vec![vec![vec![vec![0f32; k]; l]; p]; b];
        for (dst, &src) in out
            .iter_mut()
            .flatten()
            .flatten()
            .flatten()
            .zip(flat.iter())
        {
            *dst = src;
        }
        Ok(out)
    }
}

fn gather_row_positions(row: &Tensor, positions: &[usize]) -> Tensor {
    let mut v = Vec::new();
    for &p in positions {
        v.push(row.select(0, p as i64).unsqueeze(0));
    }
    Tensor::cat(&v, 0)
}
