use super::layers::{CountEmbed, create_mlp_from_dims};
use super::span_rep::SpanMarkerV0;
use crate::config::{ExtractorConfig, ModelFiles};
use crate::processor::FormattedInput;
use anyhow::Context;
use candle_core::{D, DType, Device, Result, Tensor};
use candle_nn::{Activation, Module, Sequential, VarBuilder};
use candle_transformers::models::debertav2::{Config as DebertaConfig, DebertaV2Model};

pub struct CandleExtractor {
    encoder: DebertaV2Model,
    span_rep: SpanMarkerV0,
    classifier: Sequential,
    count_pred: Sequential,
    count_embed: CountEmbed,
    hidden_size: usize,
    max_width: usize,
}

impl CandleExtractor {
    pub fn load_cpu(
        files: &ModelFiles,
        extract_config: ExtractorConfig,
        processor_vocab_size: usize,
    ) -> anyhow::Result<Self> {
        Self::load(files, extract_config, processor_vocab_size, &Device::Cpu)
    }

    pub fn load(
        files: &ModelFiles,
        extract_config: ExtractorConfig,
        processor_vocab_size: usize,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let mut encoder_config: DebertaConfig =
            serde_json::from_str(&std::fs::read_to_string(&files.encoder_config)?)
                .context("parse encoder_config")?;
        encoder_config.vocab_size = processor_vocab_size;

        let dtype = DType::F32;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&files.weights), dtype, device)
                .map_err(|e| anyhow::anyhow!("VarBuilder::from_mmaped_safetensors: {e}"))?
        };

        Self::load_raw(extract_config, encoder_config, vb).map_err(|e| anyhow::anyhow!("{e}"))
    }

    pub fn load_raw(
        config: ExtractorConfig,
        encoder_config: DebertaConfig,
        vb: VarBuilder,
    ) -> Result<Self> {
        let hidden_size = encoder_config.hidden_size;
        let max_width = config.max_width;

        let encoder = DebertaV2Model::load(vb.pp("encoder"), &encoder_config)?;

        let span_rep = SpanMarkerV0::load(
            hidden_size,
            max_width,
            vb.pp("span_rep").pp("span_rep_layer"),
        )?;

        let classifier = create_mlp_from_dims(
            hidden_size,
            &[hidden_size * 2],
            1,
            0.0,
            Activation::Relu,
            vb.pp("classifier"),
        )?;

        let count_pred = create_mlp_from_dims(
            hidden_size,
            &[hidden_size * 2],
            20,
            0.0,
            Activation::Relu,
            vb.pp("count_pred"),
        )?;

        let count_embed =
            CountEmbed::load(&config.counting_layer, hidden_size, vb.pp("count_embed"))?;

        Ok(Self {
            encoder,
            span_rep,
            classifier,
            count_pred,
            count_embed,
            hidden_size,
            max_width,
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        formatted: &FormattedInput,
    ) -> anyhow::Result<Tensor> {
        let encoder_output = self
            .encoder
            .forward(input_ids, None, Some(attention_mask.clone()))
            .map_err(candle_err)?;
        self.forward_from_encoder_output(&encoder_output, formatted)
            .map_err(candle_err)
    }

    /// NER PoC heads from encoder output `[B, seq, D]` (batch size 1).
    pub fn forward_from_encoder_output(
        &self,
        last_hidden_state: &Tensor,
        formatted: &FormattedInput,
    ) -> Result<Tensor> {
        let device = last_hidden_state.device();

        let b = last_hidden_state.dim(0)?;
        if b != 1 {
            candle_core::bail!("Only batch size 1 is supported in this PoC");
        }

        let last_hidden_state = last_hidden_state.get(0)?; // [SeqLen, D]

        let mut text_word_embs = Vec::new();
        for &pos in &formatted.text_word_first_positions {
            text_word_embs.push(last_hidden_state.get(pos)?);
        }
        let text_word_embs = Tensor::stack(&text_word_embs, 0)?; // [L, D]

        let mut schema_special_embs = Vec::new();
        for &pos in &formatted.schema_special_positions {
            schema_special_embs.push(last_hidden_state.get(pos)?);
        }
        let schema_special_embs = Tensor::stack(&schema_special_embs, 0)?; // [NumSpecial, D]

        let text_len = text_word_embs.dim(0)?;
        let mut span_indices = Vec::new();
        for i in 0..text_len {
            for w in 0..self.max_width {
                let end = (i + w).min(text_len - 1);
                span_indices.push(vec![i as u32, end as u32]);
            }
        }
        let span_indices = Tensor::new(span_indices, device)?.reshape((1, (), 2))?; // [1, S, 2]

        let span_rep = self
            .span_rep
            .forward(&text_word_embs.unsqueeze(0)?, &span_indices)?; // [1, L, max_width, D]
        let span_rep = span_rep.get(0)?; // [L, max_width, D]

        let p_emb = schema_special_embs.get(0)?; // [D]
        let count_logits = self.count_pred.forward(&p_emb.unsqueeze(0)?)?; // [1, 20]
        let pred_count = count_logits.argmax(D::Minus1)?.get(0)?.to_scalar::<u32>()? as usize;

        let num_entities = schema_special_embs.dim(0)? - 1;
        if pred_count == 0 {
            return Tensor::zeros((num_entities, text_len, self.max_width), DType::F32, device);
        }

        let e_embs = schema_special_embs.narrow(0, 1, num_entities)?; // [NumEntities, D]
        let struct_proj = self.count_embed.forward(&e_embs, pred_count)?; // [pred_count, NumEntities, D]

        let struct_proj_0 = struct_proj.get(0)?; // [NumEntities, D]

        let flat_span_rep = span_rep.reshape(((), self.hidden_size))?; // [S, D]
        let scores = flat_span_rep
            .matmul(&struct_proj_0.t()?)?
            .apply(&Activation::Sigmoid)?; // [S, NumEntities]

        let scores = scores.t()?.reshape(((), text_len, self.max_width))?;

        Ok(scores)
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn max_width(&self) -> usize {
        self.max_width
    }

    /// Encoder last hidden states `[B, seq, D]`.
    pub fn encode_sequence(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        self.encoder
            .forward(input_ids, None, Some(attention_mask.clone()))
    }

    /// First subword embedding per text word: `[L, D]` (uses batch row 0 of `last_hidden`).
    pub fn gather_text_word_embeddings(
        &self,
        last_hidden: &Tensor,
        text_word_first_positions: &[usize],
    ) -> Result<Tensor> {
        let last_hidden = last_hidden.get(0)?;
        self.gather_text_word_embeddings_row(&last_hidden, text_word_first_positions)
    }

    /// Same as [`Self::gather_text_word_embeddings`] but for one sequence `[seq, D]`.
    pub fn gather_text_word_embeddings_row(
        &self,
        last_hidden_seq: &Tensor,
        text_word_first_positions: &[usize],
    ) -> Result<Tensor> {
        let mut v = Vec::new();
        for &pos in text_word_first_positions {
            v.push(last_hidden_seq.get(pos)?);
        }
        Tensor::stack(&v, 0)
    }

    /// Gather text-word embeddings for sample `batch_idx` from encoder output `[B, seq, D]`.
    pub fn gather_text_word_embeddings_batch_idx(
        &self,
        last_hidden: &Tensor,
        batch_idx: usize,
        text_word_first_positions: &[usize],
    ) -> Result<Tensor> {
        let row = last_hidden.get(batch_idx)?;
        self.gather_text_word_embeddings_row(&row, text_word_first_positions)
    }

    /// `[L, max_width, D]` span representations (batch size 1).
    pub fn compute_span_rep(&self, text_word_embs: &Tensor) -> Result<Tensor> {
        let device = text_word_embs.device();
        let text_len = text_word_embs.dim(0)?;
        let mut span_indices = Vec::new();
        for i in 0..text_len {
            for w in 0..self.max_width {
                let end = (i + w).min(text_len - 1);
                span_indices.push(vec![i as u32, end as u32]);
            }
        }
        let span_indices = Tensor::new(span_indices, device)?.reshape((1, (), 2))?;
        let span_rep = self
            .span_rep
            .forward(&text_word_embs.unsqueeze(0)?, &span_indices)?;
        span_rep.get(0)
    }

    /// Batched span representations for variable-length `[L_i, D]` text-word embeddings.
    ///
    /// Matches Python `compute_span_rep_batched` / `_compute_span_rep_core`: pad to `max L`,
    /// mask invalid spans with `(0, 0)`, one `SpanMarkerV0` forward, then slice each row to `L_i`.
    pub fn compute_span_rep_batched(&self, token_embs_list: &[Tensor]) -> Result<Vec<Tensor>> {
        if token_embs_list.is_empty() {
            return Ok(vec![]);
        }
        let device = token_embs_list[0].device();
        let mut text_lengths = Vec::with_capacity(token_embs_list.len());
        let mut hidden = None;
        for t in token_embs_list {
            let (l, d) = t.dims2()?;
            text_lengths.push(l);
            match hidden {
                None => hidden = Some(d),
                Some(h) if h != d => candle_core::bail!("hidden dim mismatch in batch span rep"),
                _ => {}
            }
        }
        let hidden = hidden.unwrap();
        let max_text_len = *text_lengths.iter().max().unwrap();
        let batch_size = token_embs_list.len();
        let max_width = self.max_width;
        let n_spans = max_text_len * max_width;

        let mut padded = vec![0f32; batch_size * max_text_len * hidden];
        for (bi, emb) in token_embs_list.iter().enumerate() {
            let li = text_lengths[bi];
            let flat = emb.flatten_all()?.to_vec1::<f32>()?;
            for j in 0..li {
                let src = j * hidden;
                let dst = (bi * max_text_len + j) * hidden;
                padded[dst..dst + hidden].copy_from_slice(&flat[src..src + hidden]);
            }
        }
        let padded_t = Tensor::from_vec(padded, (batch_size, max_text_len, hidden), device)?;

        let mut safe_flat = vec![0u32; batch_size * n_spans * 2];
        for (b, &tl) in text_lengths.iter().enumerate().take(batch_size) {
            for i in 0..max_text_len {
                for w in 0..max_width {
                    let idx = i * max_width + w;
                    let flat_base = (b * n_spans + idx) * 2;
                    let end = i + w;
                    let valid = end < tl;
                    if valid {
                        safe_flat[flat_base] = i as u32;
                        safe_flat[flat_base + 1] = end as u32;
                    }
                }
            }
        }
        let safe_spans = Tensor::from_vec(safe_flat, (batch_size, n_spans, 2), device)?;
        let span_rep = self.span_rep.forward(&padded_t, &safe_spans)?;

        let mut out = Vec::with_capacity(batch_size);
        for (b, &tl) in text_lengths.iter().enumerate().take(batch_size) {
            let row = span_rep.get(b)?.narrow(0, 0, tl)?;
            out.push(row);
        }
        Ok(out)
    }

    /// Logits for each label embedding row `[n, D] -> [n]`.
    pub fn classifier_logits(&self, label_rows: &Tensor) -> Result<Tensor> {
        let logits = self.classifier.forward(label_rows)?;
        logits.squeeze(D::Minus1)
    }

    /// `span_rep`: `[L, K, D]`, `field_embs`: `[P, D]` → sigmoid scores `[pred_count, P, L, K]`.
    pub fn span_scores_sigmoid(
        &self,
        span_rep: &Tensor,
        field_embs: &Tensor,
        pred_count: usize,
    ) -> Result<Tensor> {
        let (l, max_w, d) = span_rep.dims3()?;
        let (p, d2) = field_embs.dims2()?;
        if d != d2 {
            candle_core::bail!("hidden dim mismatch");
        }
        let struct_proj = self.count_embed.forward(field_embs, pred_count)?;
        let span_flat = span_rep.reshape((l * max_w, d))?;
        let mut planes = Vec::new();
        for b in 0..pred_count {
            let sb = struct_proj.get(b)?;
            let scores = span_flat.matmul(&sb.t()?)?;
            let scores = scores.t()?.reshape((p, l, max_w))?;
            planes.push(scores.unsqueeze(0)?);
        }
        let stacked = Tensor::cat(&planes, 0)?;
        stacked.apply(&Activation::Sigmoid)
    }

    /// Count head: argmax over 20 logits for the `[P]` token embedding (single row `[D]`).
    pub fn count_predict(&self, p_embedding: &Tensor) -> Result<usize> {
        let count_logits = self.count_pred.forward(&p_embedding.unsqueeze(0)?)?;
        let pred_count = count_logits.argmax(D::Minus1)?.get(0)?.to_scalar::<u32>()? as usize;
        Ok(pred_count)
    }
}

fn candle_err(e: candle_core::Error) -> anyhow::Error {
    anyhow::anyhow!("{e}")
}

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

impl crate::engine::Gliner2Engine for CandleExtractor {
    type Tensor = Tensor;

    fn dup_tensor(&self, t: &Self::Tensor) -> Self::Tensor {
        t.clone()
    }

    fn hidden_size(&self) -> usize {
        CandleExtractor::hidden_size(self)
    }

    fn max_width(&self) -> usize {
        CandleExtractor::max_width(self)
    }

    fn encode_sequence(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
    ) -> anyhow::Result<Tensor> {
        CandleExtractor::encode_sequence(self, input_ids, attention_mask).map_err(candle_err)
    }

    fn gather_text_word_embeddings(
        &self,
        last_hidden: &Tensor,
        positions: &[usize],
    ) -> anyhow::Result<Tensor> {
        CandleExtractor::gather_text_word_embeddings(self, last_hidden, positions)
            .map_err(candle_err)
    }

    fn gather_text_word_embeddings_batch_idx(
        &self,
        last_hidden: &Tensor,
        batch_idx: usize,
        positions: &[usize],
    ) -> anyhow::Result<Tensor> {
        CandleExtractor::gather_text_word_embeddings_batch_idx(
            self,
            last_hidden,
            batch_idx,
            positions,
        )
        .map_err(candle_err)
    }

    fn compute_span_rep(&self, text_word_embs: &Tensor) -> anyhow::Result<Tensor> {
        CandleExtractor::compute_span_rep(self, text_word_embs).map_err(candle_err)
    }

    fn compute_span_rep_batched(&self, token_embs_list: &[Tensor]) -> anyhow::Result<Vec<Tensor>> {
        CandleExtractor::compute_span_rep_batched(self, token_embs_list).map_err(candle_err)
    }

    fn classifier_logits(&self, label_rows: &Tensor) -> anyhow::Result<Tensor> {
        CandleExtractor::classifier_logits(self, label_rows).map_err(candle_err)
    }

    fn count_predict(&self, p_embedding: &Tensor) -> anyhow::Result<usize> {
        CandleExtractor::count_predict(self, p_embedding).map_err(candle_err)
    }

    fn span_scores_sigmoid(
        &self,
        span_rep: &Tensor,
        field_embs: &Tensor,
        pred_count: usize,
    ) -> anyhow::Result<Tensor> {
        CandleExtractor::span_scores_sigmoid(self, span_rep, field_embs, pred_count)
            .map_err(candle_err)
    }

    fn single_sample_inputs(&self, input_ids: &[u32]) -> anyhow::Result<(Tensor, Tensor)> {
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
    ) -> anyhow::Result<(Tensor, Tensor)> {
        let device = Device::Cpu;
        let ids =
            Tensor::from_vec(input_ids, (batch_size, max_seq_len), &device).map_err(candle_err)?;
        let mask = Tensor::from_vec(attention_mask_i64, (batch_size, max_seq_len), &device)
            .map_err(candle_err)?;
        Ok((ids, mask))
    }

    fn batch_row_hidden(&self, hidden: &Tensor, idx: usize) -> anyhow::Result<Tensor> {
        hidden.get(idx).map_err(candle_err)
    }

    fn stack_schema_token_embeddings(
        &self,
        last_hidden_seq: &Tensor,
        positions: &[usize],
    ) -> anyhow::Result<Tensor> {
        let mut embs = Vec::new();
        for &p in positions {
            embs.push(last_hidden_seq.get(p).map_err(candle_err)?);
        }
        Tensor::stack(&embs, 0).map_err(candle_err)
    }

    fn tensor_dim0(&self, t: &Tensor) -> anyhow::Result<usize> {
        t.dim(0).map_err(candle_err)
    }

    fn tensor_narrow0(&self, t: &Tensor, start: usize, len: usize) -> anyhow::Result<Tensor> {
        t.narrow(0, start, len).map_err(candle_err)
    }

    fn tensor_index0(&self, t: &Tensor, i: usize) -> anyhow::Result<Tensor> {
        t.get(i).map_err(candle_err)
    }

    fn tensor_logits_1d(&self, logits: &Tensor) -> anyhow::Result<Vec<f32>> {
        logits.to_vec1::<f32>().map_err(candle_err)
    }

    fn tensor_span_scores_to_vec4(&self, t: &Tensor) -> anyhow::Result<Vec<Vec<Vec<Vec<f32>>>>> {
        candle_tensor_to_vec4(t).map_err(candle_err)
    }
}
