use crate::config::ExtractorConfig;
use crate::layers::{CountEmbed, create_mlp_from_dims};
use crate::processor::FormattedInput;
use crate::span_rep::SpanMarkerV0;
use candle_core::{D, DType, Result, Tensor};
use candle_nn::{Activation, Module, Sequential, VarBuilder};
use candle_transformers::models::debertav2::{Config as DebertaConfig, DebertaV2Model};

pub struct Extractor {
    encoder: DebertaV2Model,
    span_rep: SpanMarkerV0,
    classifier: Sequential,
    pub(crate) count_pred: Sequential,
    count_embed: CountEmbed,
    hidden_size: usize,
    max_width: usize,
}

impl Extractor {
    pub fn load(
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
    ) -> Result<Tensor> {
        // 1. Encode
        let encoder_output = self
            .encoder
            .forward(input_ids, None, Some(attention_mask.clone()))?;
        let last_hidden_state = encoder_output; // [B, SeqLen, D]

        let device = last_hidden_state.device();

        // 2. Extract embeddings
        // For simplicity in this PoC, we handle BatchSize = 1
        let b = last_hidden_state.dim(0)?;
        if b != 1 {
            candle_core::bail!("Only batch size 1 is supported in this PoC");
        }

        let last_hidden_state = last_hidden_state.get(0)?; // [SeqLen, D]

        // Text word embeddings
        let mut text_word_embs = Vec::new();
        for &pos in &formatted.text_word_first_positions {
            text_word_embs.push(last_hidden_state.get(pos)?);
        }
        let text_word_embs = Tensor::stack(&text_word_embs, 0)?; // [L, D]

        // Schema embeddings
        let mut schema_special_embs = Vec::new();
        for &pos in &formatted.schema_special_positions {
            schema_special_embs.push(last_hidden_state.get(pos)?);
        }
        let schema_special_embs = Tensor::stack(&schema_special_embs, 0)?; // [NumSpecial, D]

        // 3. Span Rep
        // We need to compute span indices for the text words
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

        // 4. Predict Count
        let p_emb = schema_special_embs.get(0)?; // [D] - First special token is [P]
        let count_logits = self.count_pred.forward(&p_emb.unsqueeze(0)?)?; // [1, 20]
        let pred_count = count_logits.argmax(D::Minus1)?.get(0)?.to_scalar::<u32>()? as usize;

        if pred_count == 0 {
            return Tensor::zeros((text_len, self.max_width), DType::F32, device);
        }

        // 5. Count aware project
        // schema_special_embs[1:] are [E] tokens
        let e_embs = schema_special_embs.narrow(0, 1, schema_special_embs.dim(0)? - 1)?; // [NumEntities, D]
        let struct_proj = self.count_embed.forward(&e_embs, pred_count)?; // [pred_count, NumEntities, D]

        // 6. Score
        // span_rep: [L, max_width, D]
        // struct_proj: [pred_count, NumEntities, D]
        // We want scores for entities (task 0 in our PoC)
        // scores = sigmoid(einsum("lkd,bpd->bplk", span_rep, struct_proj))
        // Here b=pred_count, p=NumEntities.
        // For NER entities task, we take b=0.

        let struct_proj_0 = struct_proj.get(0)?; // [NumEntities, D]

        // scores = sigmoid(span_rep @ struct_proj_0.T)
        // span_rep: [L*max_width, D] after flattening
        let flat_span_rep = span_rep.reshape(((), self.hidden_size))?; // [S, D]
        let scores = flat_span_rep
            .matmul(&struct_proj_0.t()?)?
            .apply(&Activation::Sigmoid)?; // [S, NumEntities]

        // Reshape back to [NumEntities, L, max_width]
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

    /// First subword embedding per text word: `[L, D]`.
    pub fn gather_text_word_embeddings(
        &self,
        last_hidden: &Tensor,
        text_word_first_positions: &[usize],
    ) -> Result<Tensor> {
        let last_hidden = last_hidden.get(0)?;
        let mut v = Vec::new();
        for &pos in text_word_first_positions {
            v.push(last_hidden.get(pos)?);
        }
        Tensor::stack(&v, 0)
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
}
