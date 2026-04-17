use super::config::{DebertaV2Config, PositionAttentionType, PositionAttentionTypes};
use crate::backends::burn::layers::{LayerNormW, LinearW};
use crate::backends::burn::weights::WeightMap;
use anyhow::Result;
use burn::prelude::*;
use burn::tensor::activation;

// ---------------------------------------------------------------------------
// Relative position helpers
// ---------------------------------------------------------------------------

/// Log-bucket relative position encoding (matches HF `make_log_bucket_position`).
fn make_log_bucket_position<B: Backend>(
    relative_pos: &Tensor<B, 2>,
    bucket_size: i64,
    max_position: i64,
) -> Tensor<B, 2> {
    let mid = bucket_size / 2;
    let sign = relative_pos.clone().sign();
    let abs_pos = relative_pos.clone().abs();
    let mid_f = mid as f64;

    // For positions beyond mid, apply log bucketing
    let log_pos = ((abs_pos.clone() / mid_f).log()
        / ((max_position - 1) as f64 / mid_f).ln()
        * (mid_f - 1.0))
    .ceil()
        + mid_f;

    let within_mid = abs_pos.clone().lower_equal_elem(mid_f);
    // Where |pos| <= mid use relative_pos, else use log_pos * sign
    let log_result = log_pos * sign;
    relative_pos.clone().mask_where(within_mid.bool_not(), log_result)
}

/// Build relative position matrix `[1, Q, K]` (matches candle/tch `build_relative_position`).
pub fn build_relative_position<B: Backend>(
    query_size: usize,
    key_size: usize,
    bucket_size: i64,
    max_position: i64,
    device: &B::Device,
) -> Tensor<B, 3> {
    // q_ids: [1, Q], k_ids: [K, 1]
    let q_ids: Tensor<B, 2> = Tensor::arange(0..query_size as i64, device)
        .float()
        .unsqueeze_dim::<2>(0); // [1, Q]
    let k_ids: Tensor<B, 2> = Tensor::arange(0..key_size as i64, device)
        .float()
        .unsqueeze_dim::<2>(1); // [K, 1]

    // rel_pos_ids = k_ids - q_ids → [K, Q]
    let mut rel_pos_ids = k_ids - q_ids;

    if bucket_size > 0 && max_position > 0 {
        rel_pos_ids = make_log_bucket_position(&rel_pos_ids, bucket_size, max_position);
    }

    // Narrow to [Q, Q] and add batch dim → [1, Q, Q]
    rel_pos_ids = rel_pos_ids.narrow(0, 0, query_size);
    rel_pos_ids.unsqueeze_dim::<3>(0)
}

// ---------------------------------------------------------------------------
// x_softmax: masked softmax
// ---------------------------------------------------------------------------

/// Softmax with mask: 1=keep, 0=masked. Masked positions get -inf before softmax, 0 after.
fn x_softmax<B: Backend>(input: &Tensor<B, 4>, mask: &Tensor<B, 4>, dim: usize) -> Tensor<B, 4> {
    let mask_f = mask.clone(); // already float [B, 1, Q, K]
    let inverse_mask = mask_f.clone().lower_elem(0.5); // bool where mask ≈ 0
    let big_neg = -1e9f32;
    let masked_input = input.clone().mask_fill(inverse_mask.clone(), big_neg);
    let result = activation::softmax(masked_input, dim);
    result.mask_fill(inverse_mask, 0.0)
}

// ---------------------------------------------------------------------------
// DebertaSelfOutput
// ---------------------------------------------------------------------------

pub struct DebertaSelfOutput<B: Backend> {
    dense: LinearW<B>,
    layer_norm: LayerNormW<B>,
}

impl<B: Backend> DebertaSelfOutput<B> {
    pub fn load(
        map: &WeightMap,
        prefix: &str,
        config: &DebertaV2Config,
        device: &B::Device,
    ) -> Result<Self> {
        let dense = LinearW::load(map, &format!("{prefix}.dense"), device)?;
        let layer_norm =
            LayerNormW::load(map, &format!("{prefix}.LayerNorm"), config.layer_norm_eps(), device)?;
        Ok(Self { dense, layer_norm })
    }

    pub fn forward(&self, hidden: &Tensor<B, 3>, input_tensor: &Tensor<B, 3>) -> Tensor<B, 3> {
        let h = self.dense.forward_3d(hidden);
        self.layer_norm.forward_3d(&(h + input_tensor.clone()))
    }
}

// ---------------------------------------------------------------------------
// DisentangledSelfAttention
// ---------------------------------------------------------------------------

pub struct DisentangledSelfAttention<B: Backend> {
    query_proj: LinearW<B>,
    key_proj: LinearW<B>,
    value_proj: LinearW<B>,
    pos_key_proj: Option<LinearW<B>>,
    pos_query_proj: Option<LinearW<B>>,
    num_attention_heads: usize,
    pos_att_type: PositionAttentionTypes,
    max_relative_positions: Option<i64>,
    position_buckets: Option<i64>,
    pos_embed_size: Option<i64>,
}

impl<B: Backend> DisentangledSelfAttention<B> {
    pub fn load(
        map: &WeightMap,
        prefix: &str,
        config: &DebertaV2Config,
        device: &B::Device,
    ) -> Result<Self> {
        let query_proj = LinearW::load(map, &format!("{prefix}.query_proj"), device)?;
        let key_proj = LinearW::load(map, &format!("{prefix}.key_proj"), device)?;
        let value_proj = LinearW::load(map, &format!("{prefix}.value_proj"), device)?;

        let pos_att_type = config.pos_att_type.clone().unwrap_or_default();
        let share_att_key = config.share_att_key.unwrap_or(false);
        let relative_attention = config.relative_attention();

        let (max_relative_positions, position_buckets, pos_embed_size, pos_key_proj, pos_query_proj) =
            if relative_attention {
                let mut mrp = config.max_relative_positions.unwrap_or(-1);
                if mrp < 1 {
                    mrp = config.max_position_embeddings as i64;
                }
                let pb = config.position_buckets.unwrap_or(-1);
                let pes = if pb > 0 { pb } else { mrp };

                let (pk, pq) = if !share_att_key {
                    let pk = if pos_att_type.has(PositionAttentionType::c2p)
                        || pos_att_type.has(PositionAttentionType::p2p)
                    {
                        Some(LinearW::load(map, &format!("{prefix}.pos_key_proj"), device)?)
                    } else {
                        None
                    };
                    let pq = if pos_att_type.has(PositionAttentionType::p2c)
                        || pos_att_type.has(PositionAttentionType::p2p)
                    {
                        Some(LinearW::load(map, &format!("{prefix}.pos_query_proj"), device)?)
                    } else {
                        None
                    };
                    (pk, pq)
                } else {
                    (None, None)
                };

                (Some(mrp), Some(pb), Some(pes), pk, pq)
            } else {
                (None, None, None, None, None)
            };

        Ok(Self {
            query_proj,
            key_proj,
            value_proj,
            pos_key_proj,
            pos_query_proj,
            num_attention_heads: config.num_attention_heads,
            pos_att_type,
            max_relative_positions,
            position_buckets,
            pos_embed_size,
        })
    }

    /// Reshape `[B, S, H]` → `[B*nhead, S, head_dim]`.
    fn transpose_for_scores(&self, x: &Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, s, _h] = x.dims();
        let head_dim = x.dims()[2] / self.num_attention_heads;
        // [B, S, nhead, head_dim] → [B, nhead, S, head_dim] → [B*nhead, S, head_dim]
        x.clone()
            .reshape([b, s, self.num_attention_heads, head_dim])
            .swap_dims(1, 2)
            .reshape([b * self.num_attention_heads, s, head_dim])
    }

    fn disentangled_att_bias(
        &self,
        query_layer: &Tensor<B, 3>,
        key_layer: &Tensor<B, 3>,
        relative_pos: &Tensor<B, 3>,
        relative_embeddings: &Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let [bh, q_len, head_dim] = query_layer.dims();
        let k_len = key_layer.dims()[1];
        let device = query_layer.device();

        let att_span = self.pos_embed_size.unwrap();

        // Slice rel embeddings to [1, 2*att_span, D] and project
        let rel_emb_sliced = relative_embeddings.clone().narrow(1, 0, (2 * att_span) as usize);

        let key_proj = self.pos_key_proj.as_ref().unwrap_or(&self.key_proj);
        let query_proj = self.pos_query_proj.as_ref().unwrap_or(&self.query_proj);

        let pos_key_layer = self.transpose_for_scores(&key_proj.forward_3d(&rel_emb_sliced));
        let pos_query_layer =
            self.transpose_for_scores(&query_proj.forward_3d(&rel_emb_sliced));

        // Repeat for each batch item
        let repeat_count = bh / self.num_attention_heads;
        let pos_key_layer = pos_key_layer.repeat_dim(0, repeat_count);
        let pos_query_layer = pos_query_layer.repeat_dim(0, repeat_count);

        let scale_factor = {
            let mut sf = 1usize;
            if self.pos_att_type.has(PositionAttentionType::c2p) {
                sf += 1;
            }
            if self.pos_att_type.has(PositionAttentionType::p2c) {
                sf += 1;
            }
            if self.pos_att_type.has(PositionAttentionType::p2p) {
                sf += 1;
            }
            sf as f64
        };

        let mut score: Tensor<B, 3> = Tensor::zeros([1, 1, 1], &device);

        // c2p attention
        if self.pos_att_type.has(PositionAttentionType::c2p)
            || self.pos_att_type.has(PositionAttentionType::p2p)
        {
            let scale = (head_dim as f64 * scale_factor).sqrt();
            // c2p_att: query_layer @ pos_key_layer^T → [bh, Q, 2*att_span]
            let c2p_att = query_layer.clone().matmul(pos_key_layer.swap_dims(1, 2));

            // c2p_pos indices: relative_pos + att_span, clamped to [0, 2*att_span-1]
            let c2p_pos = (relative_pos.clone() + att_span as f64)
                .clamp(0.0, (att_span * 2 - 1) as f64);
            // Convert to int for gather, expand to [bh, Q, K]
            let c2p_pos_int: Tensor<B, 3, Int> = c2p_pos.int();
            let c2p_pos_expanded = if c2p_pos_int.dims()[0] == 1 {
                c2p_pos_int.repeat_dim(0, bh)
            } else {
                c2p_pos_int
            };

            let c2p_att = c2p_att.gather(2, c2p_pos_expanded);
            score = score + c2p_att / scale;
        }

        // p2c attention
        if self.pos_att_type.has(PositionAttentionType::p2c) {
            let scale = (head_dim as f64 * scale_factor).sqrt();

            let r_pos = if k_len != q_len {
                build_relative_position::<B>(
                    k_len,
                    k_len,
                    self.position_buckets.unwrap_or(-1),
                    self.max_relative_positions.unwrap_or(-1),
                    &device,
                )
            } else {
                relative_pos.clone()
            };

            let p2c_pos = (r_pos.neg() + att_span as f64)
                .clamp(0.0, (att_span * 2 - 1) as f64);
            let p2c_pos_int: Tensor<B, 3, Int> = p2c_pos.int();
            let p2c_pos_expanded = if p2c_pos_int.dims()[0] == 1 {
                p2c_pos_int.repeat_dim(0, bh)
            } else {
                p2c_pos_int
            };

            // key_layer @ pos_query_layer^T → [bh, K, 2*att_span]
            let p2c_att = key_layer.clone().matmul(pos_query_layer.swap_dims(1, 2));
            let p2c_att = p2c_att.gather(2, p2c_pos_expanded);
            let p2c_att = p2c_att.swap_dims(1, 2); // [bh, K, K] → [bh, Q, K] via transpose
            score = score + p2c_att / scale;
        }

        score
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor<B, 3>,
        attention_mask: &Tensor<B, 4>,
        relative_pos: Option<&Tensor<B, 3>>,
        relative_embeddings: Option<&Tensor<B, 3>>,
    ) -> Tensor<B, 3> {
        let [b, s, _h] = hidden_states.dims();
        let head_dim = hidden_states.dims()[2] / self.num_attention_heads;
        let nhead = self.num_attention_heads;

        let query_layer = self.transpose_for_scores(&self.query_proj.forward_3d(hidden_states));
        let key_layer = self.transpose_for_scores(&self.key_proj.forward_3d(hidden_states));
        let value_layer = self.transpose_for_scores(&self.value_proj.forward_3d(hidden_states));

        // Scale factor
        let mut sf = 1usize;
        if self.pos_att_type.has(PositionAttentionType::c2p) {
            sf += 1;
        }
        if self.pos_att_type.has(PositionAttentionType::p2c) {
            sf += 1;
        }
        if self.pos_att_type.has(PositionAttentionType::p2p) {
            sf += 1;
        }
        let scale = ((head_dim * sf) as f64).sqrt();

        // Content-to-content: Q @ K^T
        let mut attn_scores = query_layer.clone().matmul(key_layer.clone().swap_dims(1, 2)) / scale;

        // Disentangled position attention
        if let (Some(rel_pos), Some(rel_embs)) = (relative_pos, relative_embeddings) {
            let rel_att =
                self.disentangled_att_bias(&query_layer, &key_layer, rel_pos, rel_embs);
            attn_scores = attn_scores + rel_att;
        }

        // Reshape to [B, nhead, S, S] for masked softmax
        let attn_scores = attn_scores.reshape([b, nhead, s, s]);
        let attn_probs = x_softmax(&attn_scores, attention_mask, 3);

        // [B, nhead, S, S] → [B*nhead, S, S] @ [B*nhead, S, head_dim]
        let attn_probs = attn_probs.reshape([b * nhead, s, s]);
        let context = attn_probs.matmul(value_layer);

        // [B*nhead, S, head_dim] → [B, nhead, S, head_dim] → [B, S, nhead, head_dim] → [B, S, H]
        let context = context
            .reshape([b, nhead, s, head_dim])
            .swap_dims(1, 2)
            .reshape([b, s, nhead * head_dim]);

        context
    }
}

// ---------------------------------------------------------------------------
// DebertaAttention (self-attention + output projection)
// ---------------------------------------------------------------------------

pub struct DebertaAttention<B: Backend> {
    self_attention: DisentangledSelfAttention<B>,
    self_output: DebertaSelfOutput<B>,
}

impl<B: Backend> DebertaAttention<B> {
    pub fn load(
        map: &WeightMap,
        prefix: &str,
        config: &DebertaV2Config,
        device: &B::Device,
    ) -> Result<Self> {
        let self_attention =
            DisentangledSelfAttention::load(map, &format!("{prefix}.self"), config, device)?;
        let self_output =
            DebertaSelfOutput::load(map, &format!("{prefix}.output"), config, device)?;
        Ok(Self {
            self_attention,
            self_output,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor<B, 3>,
        attention_mask: &Tensor<B, 4>,
        relative_pos: Option<&Tensor<B, 3>>,
        relative_embeddings: Option<&Tensor<B, 3>>,
    ) -> Tensor<B, 3> {
        let self_output = self.self_attention.forward(
            hidden_states,
            attention_mask,
            relative_pos,
            relative_embeddings,
        );
        self.self_output.forward(&self_output, hidden_states)
    }
}
