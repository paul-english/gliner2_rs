use super::weights::WeightMap;
use anyhow::Result;
use burn::prelude::*;
use burn::tensor::activation;

// ---------------------------------------------------------------------------
// LinearW — raw weight/bias linear layer
// ---------------------------------------------------------------------------

pub struct LinearW<B: Backend> {
    pub weight: Tensor<B, 2>, // [out, in]
    pub bias: Tensor<B, 1>,   // [out]
}

impl<B: Backend> LinearW<B> {
    pub fn load(map: &WeightMap, prefix: &str, device: &B::Device) -> Result<Self> {
        let weight = map.tensor2::<B>(&format!("{prefix}.weight"), device)?;
        let bias = map.tensor1::<B>(&format!("{prefix}.bias"), device)?;
        Ok(Self { weight, bias })
    }

    /// `x`: `[*, in]` → `[*, out]` (2-D).
    pub fn forward(&self, x: &Tensor<B, 2>) -> Tensor<B, 2> {
        x.clone().matmul(self.weight.clone().transpose()) + self.bias.clone().unsqueeze_dim::<2>(0)
    }

    /// 3-D variant: `[B, S, in]` → `[B, S, out]`.
    pub fn forward_3d(&self, x: &Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, s, _] = x.dims();
        let out_dim = self.weight.dims()[0];
        let flat = x.clone().reshape([b * s, x.dims()[2]]);
        let y = self.forward(&flat);
        y.reshape([b, s, out_dim])
    }
}

// ---------------------------------------------------------------------------
// LinearNoBias — weight-only linear
// ---------------------------------------------------------------------------

pub struct LinearNoBias<B: Backend> {
    pub weight: Tensor<B, 2>,
}

impl<B: Backend> LinearNoBias<B> {
    pub fn load(map: &WeightMap, prefix: &str, device: &B::Device) -> Result<Self> {
        let weight = map.tensor2::<B>(&format!("{prefix}.weight"), device)?;
        Ok(Self { weight })
    }

    pub fn forward(&self, x: &Tensor<B, 2>) -> Tensor<B, 2> {
        x.clone().matmul(self.weight.clone().transpose())
    }

    pub fn forward_3d(&self, x: &Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, s, _] = x.dims();
        let out_dim = self.weight.dims()[0];
        let flat = x.clone().reshape([b * s, x.dims()[2]]);
        let y = self.forward(&flat);
        y.reshape([b, s, out_dim])
    }
}

// ---------------------------------------------------------------------------
// LayerNormW
// ---------------------------------------------------------------------------

pub struct LayerNormW<B: Backend> {
    pub weight: Tensor<B, 1>,
    pub bias: Tensor<B, 1>,
    pub eps: f64,
}

impl<B: Backend> LayerNormW<B> {
    pub fn load(map: &WeightMap, prefix: &str, eps: f64, device: &B::Device) -> Result<Self> {
        let weight = map.tensor1::<B>(&format!("{prefix}.weight"), device)?;
        let bias = map.tensor1::<B>(&format!("{prefix}.bias"), device)?;
        Ok(Self { weight, bias, eps })
    }

    /// Layer-norm over the last dimension. Input can be 2-D or 3-D.
    pub fn forward_2d(&self, x: &Tensor<B, 2>) -> Tensor<B, 2> {
        let mean = x.clone().mean_dim(1);
        let var = x.clone().var(1);
        let normed = (x.clone() - mean) / (var + self.eps).sqrt();
        normed * self.weight.clone().unsqueeze_dim::<2>(0) + self.bias.clone().unsqueeze_dim::<2>(0)
    }

    pub fn forward_3d(&self, x: &Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, s, d] = x.dims();
        let flat = x.clone().reshape([b * s, d]);
        let y = self.forward_2d(&flat);
        y.reshape([b, s, d])
    }
}

// ---------------------------------------------------------------------------
// Embedding — raw weight matrix
// ---------------------------------------------------------------------------

pub struct EmbeddingW<B: Backend> {
    pub weight: Tensor<B, 2>, // [vocab, dim]
}

impl<B: Backend> EmbeddingW<B> {
    pub fn load(map: &WeightMap, prefix: &str, device: &B::Device) -> Result<Self> {
        let weight = map.tensor2::<B>(&format!("{prefix}.weight"), device)?;
        Ok(Self { weight })
    }

    /// Lookup `ids` `[N]` int tensor → `[N, dim]` float tensor.
    pub fn forward_1d(&self, ids: &Tensor<B, 1, Int>) -> Tensor<B, 2> {
        let dim = self.weight.dims()[1];
        let n = ids.dims()[0];
        // Expand indices to [N, dim] for gather on dim 0
        let idx: Tensor<B, 2, Int> = ids.clone().unsqueeze_dim::<2>(1).repeat_dim(1, dim);
        self.weight.clone().gather(0, idx)
    }

    /// Lookup `ids` `[B, S]` int tensor → `[B, S, dim]` float tensor.
    pub fn forward_2d(&self, ids: &Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [b, s] = ids.dims();
        let dim = self.weight.dims()[1];
        let flat = ids.clone().reshape([b * s]);
        let embs = self.forward_1d(&flat); // [b*s, dim]
        embs.reshape([b, s, dim])
    }
}

// ---------------------------------------------------------------------------
// MLP helpers
// ---------------------------------------------------------------------------

/// 2-layer MLP: Linear → ReLU → Linear, matching Python Sequential(Linear, ReLU, Linear).
pub struct Mlp2<B: Backend> {
    pub l0: LinearW<B>,
    pub l1: LinearW<B>,
}

impl<B: Backend> Mlp2<B> {
    pub fn load(map: &WeightMap, prefix: &str, device: &B::Device) -> Result<Self> {
        let l0 = LinearW::load(map, &format!("{prefix}.0"), device)?;
        // layer index 2 because index 1 is the activation
        let l1 = LinearW::load(map, &format!("{prefix}.2"), device)?;
        Ok(Self { l0, l1 })
    }

    pub fn forward(&self, x: &Tensor<B, 2>) -> Tensor<B, 2> {
        let h = activation::relu(self.l0.forward(x));
        self.l1.forward(&h)
    }

    pub fn forward_3d(&self, x: &Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, s, d] = x.dims();
        let out = self.l1.weight.dims()[0];
        let flat = x.clone().reshape([b * s, d]);
        let y = self.forward(&flat);
        y.reshape([b, s, out])
    }
}

/// 3-layer MLP: Linear → ReLU → Linear → ReLU → Linear
pub struct Mlp3<B: Backend> {
    pub l0: LinearW<B>,
    pub l1: LinearW<B>,
    pub l2: LinearW<B>,
}

impl<B: Backend> Mlp3<B> {
    pub fn load(map: &WeightMap, prefix: &str, device: &B::Device) -> Result<Self> {
        let l0 = LinearW::load(map, &format!("{prefix}.0"), device)?;
        let l1 = LinearW::load(map, &format!("{prefix}.2"), device)?;
        let l2 = LinearW::load(map, &format!("{prefix}.4"), device)?;
        Ok(Self { l0, l1, l2 })
    }

    pub fn forward(&self, x: &Tensor<B, 2>) -> Tensor<B, 2> {
        let h = activation::relu(self.l0.forward(x));
        let h = activation::relu(self.l1.forward(&h));
        self.l2.forward(&h)
    }

    pub fn forward_3d(&self, x: &Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, s, d] = x.dims();
        let out = self.l2.weight.dims()[0];
        let flat = x.clone().reshape([b * s, d]);
        let y = self.forward(&flat);
        y.reshape([b, s, out])
    }
}

/// Projection layer: Linear(in, hidden=out*4) → ReLU → Linear(hidden, out).
/// Weights at indices 0 and 3 in the Sequential (0=linear, 1=relu, 2=dropout, 3=linear).
pub struct ProjectionLayer<B: Backend> {
    l0: LinearW<B>,
    l3: LinearW<B>,
}

impl<B: Backend> ProjectionLayer<B> {
    pub fn load(map: &WeightMap, prefix: &str, device: &B::Device) -> Result<Self> {
        let l0 = LinearW::load(map, &format!("{prefix}.0"), device)?;
        let l3 = LinearW::load(map, &format!("{prefix}.3"), device)?;
        Ok(Self { l0, l3 })
    }

    pub fn forward(&self, x: &Tensor<B, 2>) -> Tensor<B, 2> {
        let h = activation::relu(self.l0.forward(x));
        self.l3.forward(&h)
    }

    pub fn forward_3d(&self, x: &Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, s, d] = x.dims();
        let out = self.l3.weight.dims()[0];
        let flat = x.clone().reshape([b * s, d]);
        let y = self.forward(&flat);
        y.reshape([b, s, out])
    }
}

// ---------------------------------------------------------------------------
// GRU (CompileSafeGRU)
// ---------------------------------------------------------------------------

pub struct GruCell<B: Backend> {
    weight_ih: Tensor<B, 2>, // [3*H, in]
    weight_hh: Tensor<B, 2>, // [3*H, H]
    bias_ih: Tensor<B, 1>,   // [3*H]
    bias_hh: Tensor<B, 1>,   // [3*H]
    hidden_size: usize,
}

impl<B: Backend> GruCell<B> {
    pub fn load(map: &WeightMap, prefix: &str, device: &B::Device) -> Result<Self> {
        let weight_ih = map.tensor2::<B>(&format!("{prefix}.weight_ih_l0"), device)?;
        let weight_hh = map.tensor2::<B>(&format!("{prefix}.weight_hh_l0"), device)?;
        let bias_ih = map.tensor1::<B>(&format!("{prefix}.bias_ih_l0"), device)?;
        let bias_hh = map.tensor1::<B>(&format!("{prefix}.bias_hh_l0"), device)?;
        let hidden_size = weight_hh.dims()[1];
        Ok(Self {
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            hidden_size,
        })
    }

    /// `x`: `[seq, batch, in]`, `h`: `[batch, H]` → output `[seq, batch, H]`.
    pub fn forward(&self, x: &Tensor<B, 3>, h_init: &Tensor<B, 2>) -> Tensor<B, 3> {
        let seq_len = x.dims()[0];
        let hs = self.hidden_size;
        let mut h = h_init.clone();
        let mut outputs = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let x_t: Tensor<B, 2> = x.clone().narrow(0, t, 1).squeeze_dim::<2>(0); // [batch, in]

            // gi = x_t @ W_ih^T + b_ih
            let gi = x_t.matmul(self.weight_ih.clone().transpose())
                + self.bias_ih.clone().unsqueeze_dim::<2>(0);
            // gh = h @ W_hh^T + b_hh
            let gh = h.clone().matmul(self.weight_hh.clone().transpose())
                + self.bias_hh.clone().unsqueeze_dim::<2>(0);

            let gi_r = gi.clone().narrow(1, 0, hs);
            let gi_z = gi.clone().narrow(1, hs, hs);
            let gi_n = gi.narrow(1, 2 * hs, hs);

            let gh_r = gh.clone().narrow(1, 0, hs);
            let gh_z = gh.clone().narrow(1, hs, hs);
            let gh_n = gh.narrow(1, 2 * hs, hs);

            let r = activation::sigmoid(gi_r + gh_r);
            let z = activation::sigmoid(gi_z + gh_z);
            let n = (gi_n + r * gh_n).tanh();

            let ones = Tensor::ones_like(&z);
            h = (ones - z.clone()) * n + z * h;
            outputs.push(h.clone());
        }

        Tensor::stack(outputs, 0)
    }
}

// ---------------------------------------------------------------------------
// CountLSTM
// ---------------------------------------------------------------------------

pub struct CountLSTM<B: Backend> {
    pos_embedding: EmbeddingW<B>,
    gru: GruCell<B>,
    projector: Mlp2<B>,
    max_count: usize,
}

impl<B: Backend> CountLSTM<B> {
    pub fn load(
        map: &WeightMap,
        prefix: &str,
        hidden_size: usize,
        max_count: usize,
        device: &B::Device,
    ) -> Result<Self> {
        let _ = hidden_size;
        let pos_embedding = EmbeddingW::load(map, &format!("{prefix}.pos_embedding"), device)?;
        let gru = GruCell::load(map, &format!("{prefix}.gru"), device)?;
        let projector = Mlp2::load(map, &format!("{prefix}.projector"), device)?;
        Ok(Self {
            pos_embedding,
            gru,
            projector,
            max_count,
        })
    }

    /// `pc_emb`: `[M, D]`, returns `[count, M, D]`.
    pub fn forward(&self, pc_emb: &Tensor<B, 2>, gold_count_val: usize) -> Tensor<B, 3> {
        let [m, d] = pc_emb.dims();
        let gold_count_val = gold_count_val.min(self.max_count);
        if gold_count_val == 0 {
            let device = pc_emb.device();
            return Tensor::zeros([0, m, d], &device);
        }

        let device = pc_emb.device();
        let count_indices: Tensor<B, 1, Int> =
            Tensor::arange(0..gold_count_val as i64, &device);
        let pos_seq = self.pos_embedding.forward_1d(&count_indices); // [count, D]

        // Expand: [count, M, D]
        let pos_seq: Tensor<B, 3> = pos_seq
            .unsqueeze_dim::<3>(1)
            .repeat_dim(1, m);

        // GRU: [count, M, D]
        let output = self.gru.forward(&pos_seq, pc_emb);

        // Concat + project: [count, M, D*2] → [count, M, D]
        let pc_expanded = pc_emb.clone().unsqueeze_dim::<3>(0).repeat_dim(0, gold_count_val);
        let combined = Tensor::cat(vec![output, pc_expanded], 2); // [count, M, D*2]
        self.projector.forward_3d(&combined)
    }
}

// ---------------------------------------------------------------------------
// TorchEncoderLayer (for DownscaledTransformer)
// ---------------------------------------------------------------------------

pub struct TorchEncoderLayer<B: Backend> {
    norm1: LayerNormW<B>,
    norm2: LayerNormW<B>,
    in_proj_weight: Tensor<B, 2>, // [3*d, d]
    in_proj_bias: Tensor<B, 1>,   // [3*d]
    out_proj: LinearW<B>,
    linear1: LinearW<B>,
    linear2: LinearW<B>,
    nhead: usize,
    head_dim: usize,
    scale: f64,
}

impl<B: Backend> TorchEncoderLayer<B> {
    pub fn load(
        map: &WeightMap,
        prefix: &str,
        d_model: usize,
        nhead: usize,
        device: &B::Device,
    ) -> Result<Self> {
        let head_dim = d_model / nhead;
        let norm1 = LayerNormW::load(map, &format!("{prefix}.norm1"), 1e-5, device)?;
        let norm2 = LayerNormW::load(map, &format!("{prefix}.norm2"), 1e-5, device)?;
        let in_proj_weight =
            map.tensor2::<B>(&format!("{prefix}.self_attn.in_proj_weight"), device)?;
        let in_proj_bias =
            map.tensor1::<B>(&format!("{prefix}.self_attn.in_proj_bias"), device)?;
        let out_proj = LinearW::load(map, &format!("{prefix}.self_attn.out_proj"), device)?;
        let linear1 = LinearW::load(map, &format!("{prefix}.linear1"), device)?;
        let linear2 = LinearW::load(map, &format!("{prefix}.linear2"), device)?;
        let scale = 1.0 / (head_dim as f64).sqrt();

        Ok(Self {
            norm1,
            norm2,
            in_proj_weight,
            in_proj_bias,
            out_proj,
            linear1,
            linear2,
            nhead,
            head_dim,
            scale,
        })
    }

    /// `x`: `[B, S, D]`.
    pub fn forward(&self, x: &Tensor<B, 3>) -> Tensor<B, 3> {
        let [b_sz, seq_len, d_model] = x.dims();
        let n = b_sz * seq_len;

        let residual = x.clone();
        let x = self.norm1.forward_3d(x);
        let x = self.mha_forward(&x, b_sz, seq_len, d_model);
        let x = x + residual;

        let residual = x.clone();
        let x = self.norm2.forward_3d(&x);
        let x_flat = x.reshape([n, d_model]);
        let x_flat = activation::relu(self.linear1.forward(&x_flat));
        let x_flat = self.linear2.forward(&x_flat);
        let x = x_flat.reshape([b_sz, seq_len, d_model]);
        x + residual
    }

    fn mha_forward(
        &self,
        x: &Tensor<B, 3>,
        b_sz: usize,
        seq_len: usize,
        d_model: usize,
    ) -> Tensor<B, 3> {
        let n = b_sz * seq_len;

        let x_flat = x.clone().reshape([n, d_model]);
        let qkv = x_flat.matmul(self.in_proj_weight.clone().transpose())
            + self.in_proj_bias.clone().unsqueeze_dim::<2>(0);
        let qkv = qkv.reshape([b_sz, seq_len, 3 * d_model]);

        let q = qkv.clone().narrow(2, 0, d_model);
        let k = qkv.clone().narrow(2, d_model, d_model);
        let v = qkv.narrow(2, 2 * d_model, d_model);

        // [B, S, nhead, head_dim] → [B, nhead, S, head_dim]
        let q = q.reshape([b_sz, seq_len, self.nhead, self.head_dim]).swap_dims(1, 2);
        let k = k.reshape([b_sz, seq_len, self.nhead, self.head_dim]).swap_dims(1, 2);
        let v = v.reshape([b_sz, seq_len, self.nhead, self.head_dim]).swap_dims(1, 2);

        // Scaled dot-product attention
        let scores = q.matmul(k.swap_dims(2, 3));
        let scores = scores * self.scale;
        let attn = activation::softmax(scores, 3);
        let out = attn.matmul(v); // [B, nhead, S, head_dim]

        // Merge heads: [B, S, nhead, head_dim] → [N, D]
        let out = out.swap_dims(1, 2).reshape([n, d_model]);
        let out = self.out_proj.forward(&out);
        out.reshape([b_sz, seq_len, d_model])
    }
}

// ---------------------------------------------------------------------------
// DownscaledTransformer
// ---------------------------------------------------------------------------

pub struct DownscaledTransformer<B: Backend> {
    in_projector: LinearW<B>,
    layers: Vec<TorchEncoderLayer<B>>,
    out_projector: Mlp3<B>,
    inner_dim: usize,
    input_size: usize,
}

impl<B: Backend> DownscaledTransformer<B> {
    pub fn load(
        map: &WeightMap,
        prefix: &str,
        input_size: usize,
        inner_dim: usize,
        device: &B::Device,
    ) -> Result<Self> {
        let in_projector = LinearW::load(map, &format!("{prefix}.in_projector"), device)?;
        let nhead = 4;
        let l0 = TorchEncoderLayer::load(
            map,
            &format!("{prefix}.transformer.layers.0"),
            inner_dim,
            nhead,
            device,
        )?;
        let l1 = TorchEncoderLayer::load(
            map,
            &format!("{prefix}.transformer.layers.1"),
            inner_dim,
            nhead,
            device,
        )?;
        let out_projector = Mlp3::load(map, &format!("{prefix}.out_projector"), device)?;
        Ok(Self {
            in_projector,
            layers: vec![l0, l1],
            out_projector,
            inner_dim,
            input_size,
        })
    }

    /// `x`: `[L, M, input_size]`.
    pub fn forward(&self, x: &Tensor<B, 3>) -> Tensor<B, 3> {
        let [l, m, _] = x.dims();
        let n = l * m;
        let original_x = x.clone();

        let h = x.clone().reshape([n, self.input_size]);
        let h = self.in_projector.forward(&h);
        let mut h = h.reshape([l, m, self.inner_dim]);
        for layer in &self.layers {
            h = layer.forward(&h);
        }
        // Concat skip connection: [L, M, inner_dim + input_size]
        let h = Tensor::cat(vec![h, original_x], 2);
        let h = h.reshape([n, self.inner_dim + self.input_size]);
        let h = self.out_projector.forward(&h);
        h.reshape([l, m, self.input_size])
    }
}

// ---------------------------------------------------------------------------
// CountLSTMv2
// ---------------------------------------------------------------------------

pub struct CountLSTMv2<B: Backend> {
    pos_embedding: EmbeddingW<B>,
    gru: GruCell<B>,
    transformer: DownscaledTransformer<B>,
    max_count: usize,
}

impl<B: Backend> CountLSTMv2<B> {
    pub fn load(
        map: &WeightMap,
        prefix: &str,
        hidden_size: usize,
        max_count: usize,
        device: &B::Device,
    ) -> Result<Self> {
        let pos_embedding = EmbeddingW::load(map, &format!("{prefix}.pos_embedding"), device)?;
        let gru = GruCell::load(map, &format!("{prefix}.gru"), device)?;
        let transformer = DownscaledTransformer::load(
            map,
            &format!("{prefix}.transformer"),
            hidden_size,
            128,
            device,
        )?;
        Ok(Self {
            pos_embedding,
            gru,
            transformer,
            max_count,
        })
    }

    /// `pc_emb`: `[M, D]`, returns `[count, M, D]`.
    pub fn forward(&self, pc_emb: &Tensor<B, 2>, gold_count_val: usize) -> Tensor<B, 3> {
        let [m, d] = pc_emb.dims();
        let gold_count_val = gold_count_val.min(self.max_count);
        if gold_count_val == 0 {
            let device = pc_emb.device();
            return Tensor::zeros([0, m, d], &device);
        }

        let device = pc_emb.device();
        let count_indices: Tensor<B, 1, Int> =
            Tensor::arange(0..gold_count_val as i64, &device);
        let pos_seq = self.pos_embedding.forward_1d(&count_indices);
        let pos_seq: Tensor<B, 3> = pos_seq.unsqueeze_dim::<3>(1).repeat_dim(1, m);

        let output = self.gru.forward(&pos_seq, pc_emb);
        let pc_broadcast = pc_emb.clone().unsqueeze_dim::<3>(0).repeat_dim(0, gold_count_val);
        let x = output + pc_broadcast;
        self.transformer.forward(&x)
    }
}

// ---------------------------------------------------------------------------
// CountEmbed (dispatch)
// ---------------------------------------------------------------------------

pub enum CountEmbed<B: Backend> {
    Lstm(CountLSTM<B>),
    LstmV2(CountLSTMv2<B>),
}

impl<B: Backend> CountEmbed<B> {
    pub fn load(
        counting_layer: &str,
        hidden_size: usize,
        map: &WeightMap,
        prefix: &str,
        device: &B::Device,
    ) -> Result<Self> {
        match counting_layer {
            "count_lstm" => Ok(Self::Lstm(CountLSTM::load(
                map,
                prefix,
                hidden_size,
                20,
                device,
            )?)),
            "count_lstm_v2" => Ok(Self::LstmV2(CountLSTMv2::load(
                map,
                prefix,
                hidden_size,
                20,
                device,
            )?)),
            other => anyhow::bail!("unsupported counting_layer: {other}"),
        }
    }

    pub fn forward(&self, pc_emb: &Tensor<B, 2>, gold_count_val: usize) -> Tensor<B, 3> {
        match self {
            Self::Lstm(m) => m.forward(pc_emb, gold_count_val),
            Self::LstmV2(m) => m.forward(pc_emb, gold_count_val),
        }
    }
}
