//! Backend abstraction for tensor inference ([`Gliner2Engine`]). Enable **`candle`** (default) and/or **`tch`** for concrete backends.

use anyhow::Result;

/// Inference backend for the GLiNER2 extract pipeline (encoder + heads + tensor helpers).
pub trait Gliner2Engine {
    type Tensor;

    /// Cheap copy for backends where [`Clone`] is unavailable (e.g. `tch::Tensor` uses `shallow_clone`).
    fn dup_tensor(&self, t: &Self::Tensor) -> Self::Tensor;

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
