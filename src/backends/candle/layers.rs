use candle_core::{D, DType, Result, Tensor, bail};
use candle_nn::{
    Activation, Embedding, LayerNorm, Linear, Module, Sequential, VarBuilder, layer_norm,
};

// Custom MLP builder to match the Python create_mlp exactly in terms of Sequential indexing
pub fn create_mlp_from_dims(
    input_dim: usize,
    intermediate_dims: &[usize],
    output_dim: usize,
    _dropout: f32,
    activation: Activation,
    vb: VarBuilder,
) -> Result<Sequential> {
    let mut seq = candle_nn::seq();
    let mut in_dim = input_dim;
    let mut layer_idx = 0;

    for &dim in intermediate_dims {
        let linear = candle_nn::linear(in_dim, dim, vb.pp(format!("{}", layer_idx)))?;
        seq = seq.add(linear);
        layer_idx += 1;

        // In the Python implementation, activation and dropout are also layers in Sequential
        seq = seq.add(activation);
        layer_idx += 1;

        in_dim = dim;
    }

    let final_linear = candle_nn::linear(in_dim, output_dim, vb.pp(format!("{}", layer_idx)))?;
    seq = seq.add(final_linear);

    Ok(seq)
}

/// Matches Python `create_projection_layer`: `Linear → ReLU → Dropout → Linear`; weights at `0` and `3`.
pub fn create_projection_layer(
    in_dim: usize,
    out_dim: usize,
    vb: VarBuilder,
) -> Result<Sequential> {
    let hidden = out_dim * 4;
    let mut seq = candle_nn::seq();
    seq = seq.add(candle_nn::linear(in_dim, hidden, vb.pp("0"))?);
    seq = seq.add(Activation::Relu);
    seq = seq.add(candle_nn::linear(hidden, out_dim, vb.pp("3"))?);
    Ok(seq)
}

/// Single PyTorch `nn.TransformerEncoderLayer` (post-norm, batch_first, ReLU FFN).
struct TorchEncoderLayer {
    norm1: LayerNorm,
    norm2: LayerNorm,
    in_proj_weight: Tensor,
    in_proj_bias: Tensor,
    out_proj: Linear,
    linear1: Linear,
    linear2: Linear,
    nhead: usize,
    head_dim: usize,
    scale: f64,
}

impl TorchEncoderLayer {
    fn load(d_model: usize, nhead: usize, dim_feedforward: usize, vb: VarBuilder) -> Result<Self> {
        if !d_model.is_multiple_of(nhead) {
            bail!("d_model {d_model} not divisible by nhead {nhead}");
        }
        let head_dim = d_model / nhead;
        let norm1 = layer_norm(d_model, 1e-5f64, vb.pp("norm1"))?;
        let norm2 = layer_norm(d_model, 1e-5f64, vb.pp("norm2"))?;
        let vb_sa = vb.pp("self_attn");
        let in_proj_weight = vb_sa.get((3 * d_model, d_model), "in_proj_weight")?;
        let in_proj_bias = vb_sa.get(3 * d_model, "in_proj_bias")?;
        let out_proj = candle_nn::linear(d_model, d_model, vb_sa.pp("out_proj"))?;
        let linear1 = candle_nn::linear(d_model, dim_feedforward, vb.pp("linear1"))?;
        let linear2 = candle_nn::linear(dim_feedforward, d_model, vb.pp("linear2"))?;
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

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b_sz, seq_len, d_model) = x.dims3()?;
        let n = b_sz * seq_len;
        let residual = x.clone();
        let x = self.norm1.forward(x)?;
        let x = self.mha_forward(&x, b_sz, seq_len, d_model)?;
        let x = x.add(&residual)?;
        let residual = x.clone();
        let x = self.norm2.forward(&x)?;
        let x = x.reshape((n, d_model))?;
        let x = self.linear1.forward(&x)?.relu()?;
        let x = self.linear2.forward(&x)?;
        let x = x.reshape((b_sz, seq_len, d_model))?;
        residual.add(&x)
    }

    fn mha_forward(
        &self,
        x: &Tensor,
        b_sz: usize,
        seq_len: usize,
        d_model: usize,
    ) -> Result<Tensor> {
        let n = b_sz * seq_len;
        let w_t = self.in_proj_weight.t()?;
        let x_flat = x.reshape((n, d_model))?;
        let mut qkv = x_flat.matmul(&w_t)?;
        let bias = self.in_proj_bias.reshape((1, 3 * d_model))?;
        qkv = qkv.broadcast_add(&bias)?;
        let qkv = qkv.reshape((b_sz, seq_len, 3 * d_model))?;
        let chunks = qkv.chunk(3, D::Minus1)?;
        let q = chunks[0].clone();
        let k = chunks[1].clone();
        let v = chunks[2].clone();

        let q = q.reshape((b_sz, seq_len, self.nhead, self.head_dim))?;
        let k = k.reshape((b_sz, seq_len, self.nhead, self.head_dim))?;
        let v = v.reshape((b_sz, seq_len, self.nhead, self.head_dim))?;

        let q = q.permute([0usize, 2, 1, 3])?;
        let k = k.permute([0usize, 2, 1, 3])?;
        let v = v.permute([0usize, 2, 1, 3])?;

        let k_t = k.transpose(D::Minus1, D::Minus2)?;
        let scores = q.matmul(&k_t)?;
        let scores = scores.broadcast_mul(&Tensor::new(self.scale as f32, scores.device())?)?;
        let attn = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let out = attn.matmul(&v)?;
        let out = out.permute([0usize, 2, 1, 3])?;
        let out = out.reshape((n, d_model))?;
        self.out_proj
            .forward(&out)?
            .reshape((b_sz, seq_len, d_model))
    }
}

/// GLiNER2 `DownscaledTransformer`: project down, 2-layer encoder, concat skip, MLP out.
pub struct DownscaledTransformer {
    in_projector: Linear,
    layers: Vec<TorchEncoderLayer>,
    out_projector: Sequential,
    inner_dim: usize,
    input_size: usize,
}

impl DownscaledTransformer {
    pub fn load(input_size: usize, inner_dim: usize, vb: VarBuilder) -> Result<Self> {
        let in_projector = candle_nn::linear(input_size, inner_dim, vb.pp("in_projector"))?;
        let nhead = 4usize;
        let dim_ff = inner_dim * 2;
        let vb_layers = vb.pp("transformer").pp("layers");
        let l0 = TorchEncoderLayer::load(inner_dim, nhead, dim_ff, vb_layers.pp("0"))?;
        let l1 = TorchEncoderLayer::load(inner_dim, nhead, dim_ff, vb_layers.pp("1"))?;
        let out_projector = create_mlp_from_dims(
            inner_dim + input_size,
            &[input_size, input_size],
            input_size,
            0.0,
            Activation::Relu,
            vb.pp("out_projector"),
        )?;
        Ok(Self {
            in_projector,
            layers: vec![l0, l1],
            out_projector,
            inner_dim,
            input_size,
        })
    }

    /// `x`: `[batch, seq, input_size]` (Python batch_first).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (l, m, insz) = x.dims3()?;
        if insz != self.input_size {
            bail!(
                "DownscaledTransformer expected input_size {}, got {}",
                self.input_size,
                insz
            );
        }
        let n = l * m;
        let original_x = x.clone();
        let h = x.reshape((n, insz))?;
        let h = self.in_projector.forward(&h)?;
        let mut h = h.reshape((l, m, self.inner_dim))?;
        for layer in &self.layers {
            h = layer.forward(&h)?;
        }
        let h = Tensor::cat(&[&h, &original_x], D::Minus1)?;
        let h = h.reshape((n, self.inner_dim + insz))?;
        let h = self.out_projector.forward(&h)?;
        h.reshape((l, m, insz))
    }
}

pub struct CountLSTMv2 {
    pos_embedding: Embedding,
    gru: CompileSafeGRU,
    transformer: DownscaledTransformer,
    max_count: usize,
}

impl CountLSTMv2 {
    pub fn load(model_hidden: usize, max_count: usize, vb: VarBuilder) -> Result<Self> {
        let pos_embedding = candle_nn::embedding(max_count, model_hidden, vb.pp("pos_embedding"))?;
        let gru = CompileSafeGRU::load(model_hidden, model_hidden, vb.pp("gru"))?;
        let transformer = DownscaledTransformer::load(model_hidden, 128, vb.pp("transformer"))?;
        Ok(Self {
            pos_embedding,
            gru,
            transformer,
            max_count,
        })
    }

    pub fn forward(&self, pc_emb: &Tensor, gold_count_val: usize) -> Result<Tensor> {
        let (m, d) = pc_emb.dims2()?;
        let gold_count_val = gold_count_val.min(self.max_count);
        if gold_count_val == 0 {
            return Tensor::zeros((0, m, d), DType::F32, pc_emb.device());
        }

        let device = pc_emb.device();
        let count_indices = Tensor::arange(0u32, gold_count_val as u32, device)?;
        let pos_seq = self.pos_embedding.forward(&count_indices)?;
        let pos_seq = pos_seq.unsqueeze(1)?.expand(&[gold_count_val, m, d])?;
        let output = self.gru.forward(&pos_seq, pc_emb.clone())?;
        let pc_broadcast = pc_emb.unsqueeze(0)?.expand(output.shape())?;
        let x = output.add(&pc_broadcast)?;
        self.transformer.forward(&x)
    }
}

pub enum CountEmbed {
    Lstm(CountLSTM),
    LstmV2(CountLSTMv2),
}

impl CountEmbed {
    pub fn load(counting_layer: &str, hidden: usize, vb: VarBuilder) -> Result<Self> {
        match counting_layer {
            "count_lstm" => Ok(CountEmbed::Lstm(CountLSTM::load(hidden, 20, vb)?)),
            "count_lstm_v2" => Ok(CountEmbed::LstmV2(CountLSTMv2::load(hidden, 20, vb)?)),
            other => bail!("unsupported counting_layer: {other} (try count_lstm or count_lstm_v2)"),
        }
    }

    pub fn forward(&self, pc_emb: &Tensor, gold_count_val: usize) -> Result<Tensor> {
        match self {
            CountEmbed::Lstm(m) => m.forward(pc_emb, gold_count_val),
            CountEmbed::LstmV2(m) => m.forward(pc_emb, gold_count_val),
        }
    }
}

pub struct CompileSafeGRU {
    weight_ih_l0: Tensor,
    weight_hh_l0: Tensor,
    bias_ih_l0: Tensor,
    bias_hh_l0: Tensor,
}

impl CompileSafeGRU {
    pub fn load(hidden_size: usize, input_size: usize, vb: VarBuilder) -> Result<Self> {
        let weight_ih_l0 = vb.get((3 * hidden_size, input_size), "weight_ih_l0")?;
        let weight_hh_l0 = vb.get((3 * hidden_size, hidden_size), "weight_hh_l0")?;
        let bias_ih_l0 = vb.get(3 * hidden_size, "bias_ih_l0")?;
        let bias_hh_l0 = vb.get(3 * hidden_size, "bias_hh_l0")?;

        Ok(Self {
            weight_ih_l0,
            weight_hh_l0,
            bias_ih_l0,
            bias_hh_l0,
        })
    }

    pub fn forward(&self, x: &Tensor, mut h: Tensor) -> Result<Tensor> {
        // x: (seq_len, batch, input_size)
        // h: (batch, hidden_size)
        let seq_len = x.dim(0)?;
        let mut outputs = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let x_t = x.get(t)?;
            let gi = x_t
                .matmul(&self.weight_ih_l0.t()?)?
                .broadcast_add(&self.bias_ih_l0)?;
            let gh = h
                .matmul(&self.weight_hh_l0.t()?)?
                .broadcast_add(&self.bias_hh_l0)?;

            let gi_chunks = gi.chunk(3, D::Minus1)?;
            let gh_chunks = gh.chunk(3, D::Minus1)?;

            let r = gi_chunks[0]
                .add(&gh_chunks[0])?
                .apply(&Activation::Sigmoid)?;
            let z = gi_chunks[1]
                .add(&gh_chunks[1])?
                .apply(&Activation::Sigmoid)?;
            let n = gi_chunks[2].add(&r.mul(&gh_chunks[2])?)?.tanh()?;

            h = z.ones_like()?.sub(&z)?.mul(&n)?.add(&z.mul(&h)?)?;
            outputs.push(h.clone());
        }

        Tensor::stack(&outputs, 0)
    }
}

pub struct CountLSTM {
    pos_embedding: Embedding,
    gru: CompileSafeGRU,
    projector: Sequential,
    max_count: usize,
}

impl CountLSTM {
    pub fn load(hidden_size: usize, max_count: usize, vb: VarBuilder) -> Result<Self> {
        let pos_embedding = candle_nn::embedding(max_count, hidden_size, vb.pp("pos_embedding"))?;
        let gru = CompileSafeGRU::load(hidden_size, hidden_size, vb.pp("gru"))?;
        let projector = create_mlp_from_dims(
            hidden_size * 2,
            &[hidden_size * 4],
            hidden_size,
            0.0,
            Activation::Relu,
            vb.pp("projector"),
        )?;

        Ok(Self {
            pos_embedding,
            gru,
            projector,
            max_count,
        })
    }

    pub fn forward(&self, pc_emb: &Tensor, gold_count_val: usize) -> Result<Tensor> {
        // pc_emb: (M, hidden_size)
        let (m, d) = pc_emb.dims2()?;
        let gold_count_val = gold_count_val.min(self.max_count);
        if gold_count_val == 0 {
            return Tensor::zeros((0, m, d), DType::F32, pc_emb.device());
        }

        let device = pc_emb.device();
        let count_indices = Tensor::arange(0u32, gold_count_val as u32, device)?;
        let pos_seq = self.pos_embedding.forward(&count_indices)?; // (gold_count_val, hidden_size)

        // Expand pos_seq: (gold_count_val, M, hidden_size)
        let pos_seq = pos_seq.unsqueeze(1)?.expand(&[gold_count_val, m, d])?;

        // Run GRU: (gold_count_val, M, hidden_size)
        let output = self.gru.forward(&pos_seq, pc_emb.clone())?;

        // Concatenate: (gold_count_val, M, hidden_size * 2)
        let pc_emb_expanded = pc_emb.unsqueeze(0)?.expand(&[gold_count_val, m, d])?;
        let combined = Tensor::cat(&[&output, &pc_emb_expanded], D::Minus1)?;

        self.projector.forward(&combined)
    }
}
