//! Backend abstraction for tensor inference ([`Gliner2Engine`]): Candle ([`crate::Extractor`]) and optional tch-rs.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

/// Inference backend for the GLiNER2 extract pipeline (encoder + heads + tensor helpers).
pub trait Gliner2Engine {
    type Tensor: Clone;

    fn hidden_size(&self) -> usize;
    fn max_width(&self) -> usize;

    fn encode_sequence(
        &self,
        input_ids: &Self::Tensor,
        attention_mask: &Self::Tensor,
    ) -> Result<Self::Tensor>;

    fn gather_text_word_embeddings(
        &self,
        last_hidden: &Self::Tensor,
        positions: &[usize],
    ) -> Result<Self::Tensor>;

    fn gather_text_word_embeddings_batch_idx(
        &self,
        last_hidden: &Self::Tensor,
        batch_idx: usize,
        positions: &[usize],
    ) -> Result<Self::Tensor>;

    fn compute_span_rep(&self, text_word_embs: &Self::Tensor) -> Result<Self::Tensor>;

    fn compute_span_rep_batched(
        &self,
        token_embs_list: &[Self::Tensor],
    ) -> Result<Vec<Self::Tensor>>;

    fn classifier_logits(&self, label_rows: &Self::Tensor) -> Result<Self::Tensor>;

    /// Argmax over the count head (20-class) for the `[P]` embedding row.
    fn count_predict(&self, p_embedding: &Self::Tensor) -> Result<usize>;

    fn span_scores_sigmoid(
        &self,
        span_rep: &Self::Tensor,
        field_embs: &Self::Tensor,
        pred_count: usize,
    ) -> Result<Self::Tensor>;

    /// `[seq]` token ids → `[1, seq]` int tensors and `[1, seq]` attention mask (ones).
    fn single_sample_inputs(&self, input_ids: &[u32]) -> Result<(Self::Tensor, Self::Tensor)>;

    fn batch_inputs(
        &self,
        input_ids: Vec<u32>,
        attention_mask_i64: Vec<i64>,
        batch_size: usize,
        max_seq_len: usize,
    ) -> Result<(Self::Tensor, Self::Tensor)>;

    /// Row `idx` of `[B, seq, D]` encoder output → `[seq, D]`.
    fn batch_row_hidden(&self, hidden: &Self::Tensor, idx: usize) -> Result<Self::Tensor>;

    /// Stack `last_hidden_seq[pos]` for schema special token positions → `[n, D]`.
    fn stack_schema_token_embeddings(
        &self,
        last_hidden_seq: &Self::Tensor,
        positions: &[usize],
    ) -> Result<Self::Tensor>;

    fn tensor_dim0(&self, t: &Self::Tensor) -> Result<usize>;

    fn tensor_narrow0(&self, t: &Self::Tensor, start: usize, len: usize) -> Result<Self::Tensor>;

    fn tensor_index0(&self, t: &Self::Tensor, i: usize) -> Result<Self::Tensor>;

    fn tensor_logits_1d(&self, logits: &Self::Tensor) -> Result<Vec<f32>>;

    fn tensor_span_scores_to_vec4(&self, t: &Self::Tensor) -> Result<Vec<Vec<Vec<Vec<f32>>>>>;
}

fn candle_err(e: candle_core::Error) -> anyhow::Error {
    anyhow::anyhow!("{e}")
}

#[allow(clippy::needless_range_loop)]
fn candle_tensor_to_vec4(t: &Tensor) -> candle_core::Result<Vec<Vec<Vec<Vec<f32>>>>> {
    let dims = t.dims();
    if dims.len() != 4 {
        candle_core::bail!("expected 4D tensor");
    }
    let b = dims[0];
    let p = dims[1];
    let l = dims[2];
    let k = dims[3];
    let flat = t.flatten_all()?.to_vec1::<f32>()?;
    let mut out = vec![vec![vec![vec![0f32; k]; l]; p]; b];
    let mut idx = 0usize;
    for bi in 0..b {
        for pi in 0..p {
            for li in 0..l {
                for ki in 0..k {
                    out[bi][pi][li][ki] = flat[idx];
                    idx += 1;
                }
            }
        }
    }
    Ok(out)
}

impl Gliner2Engine for crate::model::Extractor {
    type Tensor = Tensor;

    fn hidden_size(&self) -> usize {
        crate::model::Extractor::hidden_size(self)
    }

    fn max_width(&self) -> usize {
        crate::model::Extractor::max_width(self)
    }

    fn encode_sequence(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        crate::model::Extractor::encode_sequence(self, input_ids, attention_mask)
            .map_err(candle_err)
    }

    fn gather_text_word_embeddings(
        &self,
        last_hidden: &Tensor,
        positions: &[usize],
    ) -> Result<Tensor> {
        crate::model::Extractor::gather_text_word_embeddings(self, last_hidden, positions)
            .map_err(candle_err)
    }

    fn gather_text_word_embeddings_batch_idx(
        &self,
        last_hidden: &Tensor,
        batch_idx: usize,
        positions: &[usize],
    ) -> Result<Tensor> {
        crate::model::Extractor::gather_text_word_embeddings_batch_idx(
            self,
            last_hidden,
            batch_idx,
            positions,
        )
        .map_err(candle_err)
    }

    fn compute_span_rep(&self, text_word_embs: &Tensor) -> Result<Tensor> {
        crate::model::Extractor::compute_span_rep(self, text_word_embs).map_err(candle_err)
    }

    fn compute_span_rep_batched(&self, token_embs_list: &[Tensor]) -> Result<Vec<Tensor>> {
        crate::model::Extractor::compute_span_rep_batched(self, token_embs_list).map_err(candle_err)
    }

    fn classifier_logits(&self, label_rows: &Tensor) -> Result<Tensor> {
        crate::model::Extractor::classifier_logits(self, label_rows).map_err(candle_err)
    }

    fn count_predict(&self, p_embedding: &Tensor) -> Result<usize> {
        crate::model::Extractor::count_predict(self, p_embedding).map_err(candle_err)
    }

    fn span_scores_sigmoid(
        &self,
        span_rep: &Tensor,
        field_embs: &Tensor,
        pred_count: usize,
    ) -> Result<Tensor> {
        crate::model::Extractor::span_scores_sigmoid(self, span_rep, field_embs, pred_count)
            .map_err(candle_err)
    }

    fn single_sample_inputs(&self, input_ids: &[u32]) -> Result<(Tensor, Tensor)> {
        let device = Device::Cpu;
        let t = Tensor::new(input_ids.to_vec(), &device)
            .map_err(candle_err)?
            .unsqueeze(0)
            .map_err(candle_err)?;
        let mask = Tensor::ones(t.dims(), DType::I64, &device).map_err(candle_err)?;
        Ok((t, mask))
    }

    fn batch_inputs(
        &self,
        input_ids: Vec<u32>,
        attention_mask_i64: Vec<i64>,
        batch_size: usize,
        max_seq_len: usize,
    ) -> Result<(Tensor, Tensor)> {
        let device = Device::Cpu;
        let ids =
            Tensor::from_vec(input_ids, (batch_size, max_seq_len), &device).map_err(candle_err)?;
        let mask = Tensor::from_vec(attention_mask_i64, (batch_size, max_seq_len), &device)
            .map_err(candle_err)?;
        Ok((ids, mask))
    }

    fn batch_row_hidden(&self, hidden: &Tensor, idx: usize) -> Result<Tensor> {
        hidden.get(idx).map_err(candle_err)
    }

    fn stack_schema_token_embeddings(
        &self,
        last_hidden_seq: &Tensor,
        positions: &[usize],
    ) -> Result<Tensor> {
        let mut embs = Vec::new();
        for &p in positions {
            embs.push(last_hidden_seq.get(p).map_err(candle_err)?);
        }
        Tensor::stack(&embs, 0).map_err(candle_err)
    }

    fn tensor_dim0(&self, t: &Tensor) -> Result<usize> {
        t.dim(0).map_err(candle_err)
    }

    fn tensor_narrow0(&self, t: &Tensor, start: usize, len: usize) -> Result<Tensor> {
        t.narrow(0, start, len).map_err(candle_err)
    }

    fn tensor_index0(&self, t: &Tensor, i: usize) -> Result<Tensor> {
        t.get(i).map_err(candle_err)
    }

    fn tensor_logits_1d(&self, logits: &Tensor) -> Result<Vec<f32>> {
        logits.to_vec1::<f32>().map_err(candle_err)
    }

    fn tensor_span_scores_to_vec4(&self, t: &Tensor) -> Result<Vec<Vec<Vec<Vec<f32>>>>> {
        candle_tensor_to_vec4(t).map_err(candle_err)
    }
}
