use super::attention::{DebertaAttention, build_relative_position};
use super::config::{Activation, DebertaV2Config, NormRelEmbedType};
use super::embeddings::DebertaV2Embeddings;
use crate::backends::burn::layers::{EmbeddingW, LayerNormW, LinearW};
use crate::backends::burn::weights::WeightMap;
use anyhow::Result;
use burn::prelude::*;
use burn::tensor::activation as act;

// ---------------------------------------------------------------------------
// Intermediate + Output
// ---------------------------------------------------------------------------

fn apply_activation<B: Backend>(x: Tensor<B, 2>, act: Activation) -> Tensor<B, 2> {
    match act {
        Activation::gelu => act::gelu(x),
        Activation::relu => act::relu(x),
        Activation::swish => act::silu(x),
        Activation::tanh => x.tanh(),
        Activation::gelu_new => act::gelu(x), // approximate
    }
}

struct DebertaIntermediate<B: Backend> {
    dense: LinearW<B>,
    hidden_act: Activation,
}

impl<B: Backend> DebertaIntermediate<B> {
    fn load(
        map: &WeightMap,
        prefix: &str,
        config: &DebertaV2Config,
        device: &B::Device,
    ) -> Result<Self> {
        let dense = LinearW::load(map, &format!("{prefix}.dense"), device)?;
        Ok(Self {
            dense,
            hidden_act: config.hidden_act,
        })
    }

    fn forward(&self, hidden: &Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, s, _] = hidden.dims();
        let out_dim = self.dense.weight.dims()[0];
        let flat = hidden.clone().reshape([b * s, hidden.dims()[2]]);
        let y = apply_activation(self.dense.forward(&flat), self.hidden_act);
        y.reshape([b, s, out_dim])
    }
}

struct DebertaOutput<B: Backend> {
    dense: LinearW<B>,
    layer_norm: LayerNormW<B>,
}

impl<B: Backend> DebertaOutput<B> {
    fn load(
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

    fn forward(&self, hidden: &Tensor<B, 3>, input_tensor: &Tensor<B, 3>) -> Tensor<B, 3> {
        let h = self.dense.forward_3d(hidden);
        self.layer_norm.forward_3d(&(input_tensor.clone() + h))
    }
}

// ---------------------------------------------------------------------------
// DebertaV2Layer
// ---------------------------------------------------------------------------

struct DebertaV2Layer<B: Backend> {
    attention: DebertaAttention<B>,
    intermediate: DebertaIntermediate<B>,
    output: DebertaOutput<B>,
}

impl<B: Backend> DebertaV2Layer<B> {
    fn load(
        map: &WeightMap,
        prefix: &str,
        config: &DebertaV2Config,
        device: &B::Device,
    ) -> Result<Self> {
        let attention = DebertaAttention::load(map, &format!("{prefix}.attention"), config, device)?;
        let intermediate =
            DebertaIntermediate::load(map, &format!("{prefix}.intermediate"), config, device)?;
        let output = DebertaOutput::load(map, &format!("{prefix}.output"), config, device)?;
        Ok(Self {
            attention,
            intermediate,
            output,
        })
    }

    fn forward(
        &self,
        hidden: &Tensor<B, 3>,
        attention_mask: &Tensor<B, 4>,
        relative_pos: Option<&Tensor<B, 3>>,
        relative_embeddings: Option<&Tensor<B, 3>>,
    ) -> Tensor<B, 3> {
        let attn_out =
            self.attention
                .forward(hidden, attention_mask, relative_pos, relative_embeddings);
        let intermediate_out = self.intermediate.forward(&attn_out);
        self.output.forward(&intermediate_out, &attn_out)
    }
}

// ---------------------------------------------------------------------------
// ConvLayer (optional, after first encoder layer)
// ---------------------------------------------------------------------------

struct ConvLayer<B: Backend> {
    // 1-D convolution implemented as grouped linear for simplicity.
    conv_weight: Tensor<B, 3>, // [out_ch, in_ch/groups, kernel]
    conv_bias: Tensor<B, 1>,
    layer_norm: LayerNormW<B>,
    conv_act: Activation,
    kernel_size: usize,
    groups: usize,
}

impl<B: Backend> ConvLayer<B> {
    fn load(
        map: &WeightMap,
        prefix: &str,
        config: &DebertaV2Config,
        device: &B::Device,
    ) -> Result<Self> {
        let conv_weight = map.tensor3::<B>(&format!("{prefix}.conv.weight"), device)?;
        let conv_bias = map.tensor1::<B>(&format!("{prefix}.conv.bias"), device)?;
        let layer_norm =
            LayerNormW::load(map, &format!("{prefix}.LayerNorm"), config.layer_norm_eps(), device)?;
        let conv_act = config.conv_act.unwrap_or(Activation::tanh);
        let kernel_size = config.conv_kernel_size.unwrap_or(3);
        let groups = config.conv_groups.unwrap_or(1);
        Ok(Self {
            conv_weight,
            conv_bias,
            layer_norm,
            conv_act,
            kernel_size,
            groups,
        })
    }

    /// Manual 1-D convolution: `[B, S, D]` → `[B, S, D]`.
    fn forward(
        &self,
        hidden: &Tensor<B, 3>,
        residual: &Tensor<B, 3>,
        input_mask: &Tensor<B, 2, Int>,
    ) -> Tensor<B, 3> {
        let [b, s, d] = hidden.dims();
        let pad = (self.kernel_size - 1) / 2;
        let device = hidden.device();

        // Pad input: [B, S, D] → [B, S + 2*pad, D]
        let padded = if pad > 0 {
            let zeros = Tensor::<B, 3>::zeros([b, pad, d], &device);
            Tensor::cat(vec![zeros.clone(), hidden.clone(), zeros], 1)
        } else {
            hidden.clone()
        };

        // Simple grouped 1-D conv via unfolding
        // conv_weight: [out_ch, in_ch/groups, kernel]
        let [out_ch, ch_per_group, _k] = self.conv_weight.dims();
        let group_size = d / self.groups;

        let mut output_groups = Vec::new();
        for g in 0..self.groups {
            let in_start = g * group_size;
            let out_start = g * (out_ch / self.groups);
            let n_out = out_ch / self.groups;

            // Extract input channels for this group: [B, S+2p, group_size]
            let inp = padded.clone().narrow(2, in_start, group_size);

            // Unfold: collect windows → [B*S, group_size * kernel]
            let mut windows = Vec::with_capacity(s);
            for i in 0..s {
                let window = inp.clone().narrow(1, i, self.kernel_size); // [B, K, gs]
                windows.push(window.reshape([b, group_size * self.kernel_size]));
            }
            // Stack to [B, S, gs*K], then reshape to [B*S, gs*K] for 2D matmul
            let unfolded: Tensor<B, 3> = Tensor::stack(windows, 1);
            let unfolded_flat: Tensor<B, 2> = unfolded.reshape([b * s, ch_per_group * self.kernel_size]);

            // Weight for this group: [n_out, ch_per_group, K] → [n_out, ch_per_group*K]
            let w: Tensor<B, 2> = self
                .conv_weight
                .clone()
                .narrow(0, out_start, n_out)
                .reshape([n_out, ch_per_group * self.kernel_size]);
            // [B*S, gs*K] @ [gs*K, n_out] → [B*S, n_out] → [B, S, n_out]
            let conv_out: Tensor<B, 3> = unfolded_flat.matmul(w.transpose()).reshape([b, s, n_out]);
            output_groups.push(conv_out);
        }

        let mut out: Tensor<B, 3> = Tensor::cat(output_groups, 2); // [B, S, D]
        // Add bias: [D] → [1, 1, D] for broadcast
        let bias_3d: Tensor<B, 3> = self.conv_bias.clone()
            .unsqueeze_dim::<2>(0)
            .unsqueeze_dim::<3>(0);
        out = out + bias_3d;

        // Mask: [B, S] → [B, S, 1] → bool
        let mask_f: Tensor<B, 3> = input_mask.clone().float().unsqueeze_dim::<3>(2);
        let reverse_mask: Tensor<B, 3, burn::tensor::Bool> = mask_f.clone().lower_elem(0.5);
        out = out.mask_fill(reverse_mask, 0.0);

        // Activation
        out = match self.conv_act {
            Activation::tanh => out.tanh(),
            Activation::relu => act::relu(out),
            Activation::gelu => act::gelu(out),
            Activation::swish => act::silu(out),
            Activation::gelu_new => act::gelu(out),
        };

        let layer_norm_input: Tensor<B, 3> = residual.clone() + out;
        let output = self.layer_norm.forward_3d(&layer_norm_input);
        let mask_f2: Tensor<B, 3> = input_mask.clone().float().unsqueeze_dim::<3>(2);
        output * mask_f2
    }
}

// ---------------------------------------------------------------------------
// DebertaV2Encoder
// ---------------------------------------------------------------------------

struct DebertaV2Encoder<B: Backend> {
    layers: Vec<DebertaV2Layer<B>>,
    rel_embeddings: Option<EmbeddingW<B>>,
    layer_norm: Option<LayerNormW<B>>,
    conv: Option<ConvLayer<B>>,
    max_relative_positions: Option<i64>,
    position_buckets: Option<i64>,
}

impl<B: Backend> DebertaV2Encoder<B> {
    fn load(
        map: &WeightMap,
        prefix: &str,
        config: &DebertaV2Config,
        device: &B::Device,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(DebertaV2Layer::load(
                map,
                &format!("{prefix}.layer.{i}"),
                config,
                device,
            )?);
        }

        let (rel_embeddings, max_relative_positions, position_buckets) =
            if config.relative_attention() {
                let mut mrp = config.max_relative_positions.unwrap_or(-1);
                if mrp < 1 {
                    mrp = config.max_position_embeddings as i64;
                }
                let pb = config.position_buckets.unwrap_or(-1);
                let rel_emb =
                    EmbeddingW::load(map, &format!("{prefix}.rel_embeddings"), device)?;
                (Some(rel_emb), Some(mrp), Some(pb))
            } else {
                (None, None, None)
            };

        let layer_norm =
            if config
                .norm_rel_ebd
                .clone()
                .unwrap_or_default()
                .has(NormRelEmbedType::layer_norm)
            {
                Some(LayerNormW::load(
                    map,
                    &format!("{prefix}.LayerNorm"),
                    1e-7,
                    device,
                )?)
            } else {
                None
            };

        let conv = if config.conv_kernel_size.unwrap_or(0) > 0 {
            Some(ConvLayer::load(map, &format!("{prefix}.conv"), config, device)?)
        } else {
            None
        };

        Ok(Self {
            layers,
            rel_embeddings,
            layer_norm,
            conv,
            max_relative_positions,
            position_buckets,
        })
    }

    fn get_rel_embedding(&self) -> Option<Tensor<B, 3>> {
        self.rel_embeddings.as_ref().map(|emb| {
            let w: Tensor<B, 3> = emb.weight.clone().unsqueeze_dim::<3>(0); // [1, vocab, D]
            if let Some(ln) = &self.layer_norm {
                ln.forward_3d(&w)
            } else {
                w
            }
        })
    }

    /// Broadcast `[B, S]` int mask → `[B, 1, S, S]` float attention mask.
    fn get_attention_mask(attention_mask: &Tensor<B, 2, Int>) -> Tensor<B, 4> {
        let m = attention_mask.clone().float(); // [B, S]
        // [B, 1, 1, S] * [B, 1, S, 1] → [B, 1, S, S]
        let left: Tensor<B, 4> = m.clone().unsqueeze_dim::<3>(1).unsqueeze_dim::<4>(2); // [B, 1, 1, S]
        let right: Tensor<B, 4> = m.unsqueeze_dim::<3>(1).unsqueeze_dim::<4>(3); // [B, 1, S, 1]
        left * right
    }

    fn forward(
        &self,
        hidden_states: &Tensor<B, 3>,
        attention_mask: &Tensor<B, 2, Int>,
    ) -> Tensor<B, 3> {
        let s = hidden_states.dims()[1];
        let device = hidden_states.device();

        let attn_mask_4d = Self::get_attention_mask(attention_mask);

        let relative_pos = if self.rel_embeddings.is_some() {
            Some(build_relative_position::<B>(
                s,
                s,
                self.position_buckets.unwrap(),
                self.max_relative_positions.unwrap(),
                &device,
            ))
        } else {
            None
        };
        let relative_embeddings = self.get_rel_embedding();

        let mut output = hidden_states.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            output = layer.forward(
                &output,
                &attn_mask_4d,
                relative_pos.as_ref(),
                relative_embeddings.as_ref(),
            );
            if i == 0 {
                if let Some(conv) = &self.conv {
                    output = conv.forward(hidden_states, &output, attention_mask);
                }
            }
        }

        output
    }
}

// ---------------------------------------------------------------------------
// DebertaV2Model (top-level)
// ---------------------------------------------------------------------------

pub struct DebertaV2Model<B: Backend> {
    embeddings: DebertaV2Embeddings<B>,
    encoder: DebertaV2Encoder<B>,
}

impl<B: Backend> DebertaV2Model<B> {
    pub fn load(
        map: &WeightMap,
        prefix: &str,
        config: &DebertaV2Config,
        device: &B::Device,
    ) -> Result<Self> {
        let embeddings =
            DebertaV2Embeddings::load(map, &format!("{prefix}.embeddings"), config, device)?;
        let encoder =
            DebertaV2Encoder::load(map, &format!("{prefix}.encoder"), config, device)?;
        Ok(Self {
            embeddings,
            encoder,
        })
    }

    /// `input_ids`: `[B, S]` int, `attention_mask`: `[B, S]` int → `[B, S, D]` float.
    pub fn forward(
        &self,
        input_ids: &Tensor<B, 2, Int>,
        attention_mask: &Tensor<B, 2, Int>,
    ) -> Tensor<B, 3> {
        let embedding_output = self.embeddings.forward(input_ids, attention_mask);
        self.encoder.forward(&embedding_output, attention_mask)
    }
}
