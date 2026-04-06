//! GLiNER heads on `tch::Tensor`, loaded from the same `safetensors` keys as the Candle backend.

use super::weights::LinearW;
use crate::config::ExtractorConfig;
use anyhow::{Context, Result, bail};
use std::collections::HashMap;
use tch::{Device, Kind, Tensor};

const LN_EPS: f64 = 1e-5;

fn layer_norm(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Tensor {
    let mean = x.mean_dim(&[-1i64][..], true, Kind::Float);
    let xc = x - mean;
    let var = (&xc * &xc).mean_dim(&[-1i64][..], true, Kind::Float);
    let norm = &xc * (var + LN_EPS).rsqrt();
    norm * weight + bias
}

struct LayerNormW {
    weight: Tensor,
    bias: Tensor,
}

impl LayerNormW {
    fn from_map(map: &HashMap<String, Tensor>, prefix: &str) -> Result<Self> {
        let wkey = format!("{prefix}.weight");
        let bkey = format!("{prefix}.bias");
        Ok(Self {
            weight: map.get(&wkey).with_context(|| wkey)?.shallow_clone(),
            bias: map.get(&bkey).with_context(|| bkey)?.shallow_clone(),
        })
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        layer_norm(x, &self.weight, &self.bias)
    }
}

struct ProjectionT {
    l0: LinearW,
    l1: LinearW,
}

impl ProjectionT {
    fn from_map(map: &HashMap<String, Tensor>, prefix: &str) -> Result<Self> {
        Ok(Self {
            l0: LinearW::from_map(map, &format!("{prefix}.0"))?,
            l1: LinearW::from_map(map, &format!("{prefix}.3"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        self.l1.forward(&self.l0.forward(x).relu())
    }
}

pub struct TchSpanMarkerV0 {
    project_start: ProjectionT,
    project_end: ProjectionT,
    out_project: ProjectionT,
    max_width: usize,
}

impl TchSpanMarkerV0 {
    pub fn load(map: &HashMap<String, Tensor>, max_width: usize, prefix: &str) -> Result<Self> {
        Ok(Self {
            project_start: ProjectionT::from_map(map, &format!("{prefix}.project_start"))?,
            project_end: ProjectionT::from_map(map, &format!("{prefix}.project_end"))?,
            out_project: ProjectionT::from_map(map, &format!("{prefix}.out_project"))?,
            max_width,
        })
    }

    pub fn forward(&self, h: &Tensor, span_idx: &Tensor) -> Tensor {
        let sz = h.size();
        let b = sz[0];
        let l = sz[1];
        let d = sz[2];

        let start_rep = self.project_start.forward(h);
        let end_rep = self.project_end.forward(h);

        let starts = span_idx.select(2, 0).to_kind(Kind::Int64);
        let ends = span_idx.select(2, 1).to_kind(Kind::Int64);
        let s = starts.size()[1];

        let expanded_s = starts.unsqueeze(2).expand([b, s, d], false);
        let expanded_e = ends.unsqueeze(2).expand([b, s, d], false);

        let start_span_rep = start_rep.gather(1, &expanded_s, false);
        let end_span_rep = end_rep.gather(1, &expanded_e, false);

        let cat = Tensor::cat(&[&start_span_rep, &end_span_rep], -1).relu();
        let out = self.out_project.forward(&cat);
        out.reshape([b, l, self.max_width as i64, d])
    }
}

pub struct TchMlp3 {
    pub l0: LinearW,
    pub l1: LinearW,
    pub l2: LinearW,
}

impl TchMlp3 {
    pub fn from_map(map: &HashMap<String, Tensor>, prefix: &str) -> Result<Self> {
        Ok(Self {
            l0: LinearW::from_map(map, &format!("{prefix}.0"))?,
            l1: LinearW::from_map(map, &format!("{prefix}.2"))?,
            l2: LinearW::from_map(map, &format!("{prefix}.4"))?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let x = self.l0.forward(x).relu();
        let x = self.l1.forward(&x).relu();
        self.l2.forward(&x)
    }
}

struct GrucellT {
    w_ih: Tensor,
    w_hh: Tensor,
    b_ih: Tensor,
    b_hh: Tensor,
}

impl GrucellT {
    fn from_map(
        map: &HashMap<String, Tensor>,
        prefix: &str,
        hidden: i64,
        input: i64,
    ) -> Result<Self> {
        let w_ih = map
            .get(&format!("{prefix}.weight_ih_l0"))
            .with_context(|| format!("{prefix}.weight_ih_l0"))?
            .shallow_clone();
        let w_hh = map
            .get(&format!("{prefix}.weight_hh_l0"))
            .with_context(|| format!("{prefix}.weight_hh_l0"))?
            .shallow_clone();
        let b_ih = map
            .get(&format!("{prefix}.bias_ih_l0"))
            .with_context(|| format!("{prefix}.bias_ih_l0"))?
            .shallow_clone();
        let b_hh = map
            .get(&format!("{prefix}.bias_hh_l0"))
            .with_context(|| format!("{prefix}.bias_hh_l0"))?
            .shallow_clone();
        let (wih0, wih1) = w_ih.size2().unwrap();
        if wih0 != 3 * hidden || wih1 != input {
            bail!("GRU weight_ih shape mismatch");
        }
        Ok(Self {
            w_ih,
            w_hh,
            b_ih,
            b_hh,
        })
    }

    /// `x`: `[seq, batch, input]`, `h0`: `[batch, hidden]` → `[seq, batch, hidden]`.
    fn forward(&self, x: &Tensor, mut h: Tensor) -> Tensor {
        let seq_len = x.size()[0];
        let mut outs = Vec::with_capacity(seq_len as usize);
        let w_ih_t = self.w_ih.transpose(0, 1);
        let w_hh_t = self.w_hh.transpose(0, 1);
        for t in 0..seq_len {
            let x_t = x.select(0, t);
            let gi = x_t.matmul(&w_ih_t) + &self.b_ih;
            let gh = h.matmul(&w_hh_t) + &self.b_hh;
            let chunks_i = gi.chunk(3, -1);
            let chunks_h = gh.chunk(3, -1);
            let r = (chunks_i[0].shallow_clone() + &chunks_h[0]).sigmoid();
            let z = (chunks_i[1].shallow_clone() + &chunks_h[1]).sigmoid();
            let n = (chunks_i[2].shallow_clone() + &(r * &chunks_h[2])).tanh();
            let one = z.ones_like();
            h = (one - &z) * &n + &z * &h;
            outs.push(h.shallow_clone());
        }
        Tensor::stack(&outs, 0)
    }
}

pub(crate) struct TchCountLstm {
    pos_embedding: Tensor,
    gru: GrucellT,
    projector: TchMlp3,
    max_count: usize,
}

impl TchCountLstm {
    fn load(
        map: &HashMap<String, Tensor>,
        hidden: usize,
        max_count: usize,
        prefix: &str,
    ) -> Result<Self> {
        let emb = map
            .get(&format!("{prefix}.pos_embedding.weight"))
            .with_context(|| format!("{prefix}.pos_embedding.weight"))?
            .shallow_clone();
        let gru = GrucellT::from_map(map, &format!("{prefix}.gru"), hidden as i64, hidden as i64)?;
        let projector = TchMlp3::from_map(map, &format!("{prefix}.projector"))?;
        Ok(Self {
            pos_embedding: emb,
            gru,
            projector,
            max_count,
        })
    }

    fn forward(&self, pc_emb: &Tensor, gold_count_val: usize) -> Tensor {
        let (m, d) = pc_emb.size2().unwrap();
        let g = gold_count_val.min(self.max_count);
        if g == 0 {
            let dev = pc_emb.device();
            return Tensor::zeros([0, m, d], (Kind::Float, dev));
        }
        let dev = pc_emb.device();
        let idx = Tensor::arange_start(0i64, g as i64, (Kind::Int64, dev));
        let pos_seq = self.pos_embedding.index_select(0, &idx);
        let pos_seq = pos_seq.unsqueeze(1).expand([g as i64, m, d], false);
        let output = self.gru.forward(&pos_seq, pc_emb.shallow_clone());
        let pc_exp = pc_emb.unsqueeze(0).expand(output.size().as_slice(), false);
        let combined = Tensor::cat(&[&output, &pc_exp], -1);
        self.projector.forward(&combined)
    }
}

struct TorchEncoderLayerT {
    norm1: LayerNormW,
    norm2: LayerNormW,
    in_proj_weight: Tensor,
    in_proj_bias: Tensor,
    out_proj: LinearW,
    linear1: LinearW,
    linear2: LinearW,
    nhead: i64,
    head_dim: i64,
    scale: f64,
}

impl TorchEncoderLayerT {
    fn load(
        map: &HashMap<String, Tensor>,
        d_model: i64,
        nhead: i64,
        _dim_ff: i64,
        prefix: &str,
    ) -> Result<Self> {
        if d_model % nhead != 0 {
            bail!("d_model not divisible by nhead");
        }
        let head_dim = d_model / nhead;
        let norm1 = LayerNormW::from_map(map, &format!("{prefix}.norm1"))?;
        let norm2 = LayerNormW::from_map(map, &format!("{prefix}.norm2"))?;
        let vb_sa = format!("{prefix}.self_attn");
        let in_proj_weight = map
            .get(&format!("{vb_sa}.in_proj_weight"))
            .with_context(|| format!("{vb_sa}.in_proj_weight"))?
            .shallow_clone();
        let in_proj_bias = map
            .get(&format!("{vb_sa}.in_proj_bias"))
            .with_context(|| format!("{vb_sa}.in_proj_bias"))?
            .shallow_clone();
        let out_proj = LinearW::from_map(map, &format!("{vb_sa}.out_proj"))?;
        let linear1 = LinearW::from_map(map, &format!("{prefix}.linear1"))?;
        let linear2 = LinearW::from_map(map, &format!("{prefix}.linear2"))?;
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

    fn mha_forward(&self, x: &Tensor, b_sz: i64, seq_len: i64, d_model: i64) -> Tensor {
        let n = b_sz * seq_len;
        let w_t = self.in_proj_weight.transpose(0, 1);
        let x_flat = x.reshape([n, d_model]);
        let qkv = x_flat.matmul(&w_t) + self.in_proj_bias.reshape([1, 3 * d_model]);
        let qkv = qkv.reshape([b_sz, seq_len, 3 * d_model]);
        let chunks = qkv.chunk(3, -1);
        let q = chunks[0].shallow_clone();
        let k = chunks[1].shallow_clone();
        let v = chunks[2].shallow_clone();

        let q = q
            .reshape([b_sz, seq_len, self.nhead, self.head_dim])
            .permute([0, 2, 1, 3]);
        let k = k
            .reshape([b_sz, seq_len, self.nhead, self.head_dim])
            .permute([0, 2, 1, 3]);
        let v = v
            .reshape([b_sz, seq_len, self.nhead, self.head_dim])
            .permute([0, 2, 1, 3]);

        let k_t = k.transpose(-1, -2);
        let mut scores = q.matmul(&k_t);
        scores *= self.scale;
        let attn = scores.softmax(-1, Kind::Float);
        let out = attn.matmul(&v);
        let out = out.permute([0, 2, 1, 3]).reshape([n, d_model]);
        self.out_proj
            .forward(&out)
            .reshape([b_sz, seq_len, d_model])
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let b_sz = x.size()[0];
        let seq_len = x.size()[1];
        let d_model = x.size()[2];
        let n = b_sz * seq_len;
        let residual = x.shallow_clone();
        let x = self.norm1.forward(x);
        let x = self.mha_forward(&x, b_sz, seq_len, d_model);
        let x = x + residual;
        let residual = x.shallow_clone();
        let x = self.norm2.forward(&x);
        let x = x.reshape([n, d_model]);
        let x = self.linear2.forward(&self.linear1.forward(&x).relu());
        let x = x.reshape([b_sz, seq_len, d_model]);
        x + residual
    }
}

struct DownscaledTransformerT {
    in_projector: LinearW,
    layers: Vec<TorchEncoderLayerT>,
    out_projector: TchMlp3,
    inner_dim: i64,
    input_size: i64,
}

impl DownscaledTransformerT {
    fn load(
        map: &HashMap<String, Tensor>,
        input_size: i64,
        inner_dim: i64,
        prefix: &str,
    ) -> Result<Self> {
        let in_projector = LinearW::from_map(map, &format!("{prefix}.in_projector"))?;
        let nhead = 4i64;
        let dim_ff = inner_dim * 2;
        let vb_layers = format!("{prefix}.transformer.layers");
        let l0 =
            TorchEncoderLayerT::load(map, inner_dim, nhead, dim_ff, &format!("{vb_layers}.0"))?;
        let l1 =
            TorchEncoderLayerT::load(map, inner_dim, nhead, dim_ff, &format!("{vb_layers}.1"))?;
        let out_projector = TchMlp3::from_map(map, &format!("{prefix}.out_projector"))?;
        Ok(Self {
            in_projector,
            layers: vec![l0, l1],
            out_projector,
            inner_dim,
            input_size,
        })
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let l = x.size()[0];
        let m = x.size()[1];
        let insz = x.size()[2];
        assert_eq!(insz, self.input_size);
        let n = l * m;
        let original_x = x.shallow_clone();
        let h = x.reshape([n, insz]);
        let h = self.in_projector.forward(&h);
        let mut h = h.reshape([l, m, self.inner_dim]);
        for layer in &self.layers {
            h = layer.forward(&h);
        }
        let h = Tensor::cat(&[&h, &original_x], -1);
        let h = h.reshape([n, self.inner_dim + insz]);
        self.out_projector.forward(&h).reshape([l, m, insz])
    }
}

pub(crate) struct TchCountLstmV2 {
    pos_embedding: Tensor,
    gru: GrucellT,
    transformer: DownscaledTransformerT,
    max_count: usize,
}

impl TchCountLstmV2 {
    fn load(
        map: &HashMap<String, Tensor>,
        hidden: usize,
        max_count: usize,
        prefix: &str,
    ) -> Result<Self> {
        let emb = map
            .get(&format!("{prefix}.pos_embedding.weight"))
            .with_context(|| format!("{prefix}.pos_embedding.weight"))?
            .shallow_clone();
        let gru = GrucellT::from_map(map, &format!("{prefix}.gru"), hidden as i64, hidden as i64)?;
        let transformer = DownscaledTransformerT::load(
            map,
            hidden as i64,
            128,
            &format!("{prefix}.transformer"),
        )?;
        Ok(Self {
            pos_embedding: emb,
            gru,
            transformer,
            max_count,
        })
    }

    fn forward(&self, pc_emb: &Tensor, gold_count_val: usize) -> Tensor {
        let (m, d) = pc_emb.size2().unwrap();
        let g = gold_count_val.min(self.max_count);
        if g == 0 {
            let dev = pc_emb.device();
            return Tensor::zeros([0, m, d], (Kind::Float, dev));
        }
        let dev = pc_emb.device();
        let idx = Tensor::arange_start(0i64, g as i64, (Kind::Int64, dev));
        let pos_seq = self.pos_embedding.index_select(0, &idx);
        let pos_seq = pos_seq.unsqueeze(1).expand([g as i64, m, d], false);
        let output = self.gru.forward(&pos_seq, pc_emb.shallow_clone());
        let pc_broadcast = pc_emb.unsqueeze(0).expand(output.size().as_slice(), false);
        let x = output + pc_broadcast;
        self.transformer.forward(&x)
    }
}

pub enum TchCountEmbed {
    Lstm(TchCountLstm),
    LstmV2(TchCountLstmV2),
}

impl TchCountEmbed {
    pub fn load(
        map: &HashMap<String, Tensor>,
        counting_layer: &str,
        hidden: usize,
        prefix: &str,
    ) -> Result<Self> {
        match counting_layer {
            "count_lstm" => Ok(Self::Lstm(TchCountLstm::load(map, hidden, 20, prefix)?)),
            "count_lstm_v2" => Ok(Self::LstmV2(TchCountLstmV2::load(map, hidden, 20, prefix)?)),
            other => bail!("unsupported counting_layer: {other}"),
        }
    }

    pub fn forward(&self, pc_emb: &Tensor, gold_count_val: usize) -> Tensor {
        match self {
            TchCountEmbed::Lstm(m) => m.forward(pc_emb, gold_count_val),
            TchCountEmbed::LstmV2(m) => m.forward(pc_emb, gold_count_val),
        }
    }
}

pub struct TchHeads {
    pub hidden_size: usize,
    pub max_width: usize,
    span_rep: TchSpanMarkerV0,
    classifier: TchMlp3,
    count_pred: TchMlp3,
    count_embed: TchCountEmbed,
}

impl TchHeads {
    pub fn load(
        map: &HashMap<String, Tensor>,
        _device: Device,
        cfg: &ExtractorConfig,
    ) -> Result<Self> {
        let hidden = {
            let w = map
                .get("classifier.0.weight")
                .context("classifier.0.weight for hidden_size")?;
            w.size()[1] as usize
        };
        let max_width = cfg.max_width;
        let span_rep = TchSpanMarkerV0::load(map, max_width, "span_rep.span_rep_layer")?;
        let classifier = TchMlp3::from_map(map, "classifier")?;
        let count_pred = TchMlp3::from_map(map, "count_pred")?;
        let count_embed = TchCountEmbed::load(map, &cfg.counting_layer, hidden, "count_embed")?;
        Ok(Self {
            hidden_size: hidden,
            max_width,
            span_rep,
            classifier,
            count_pred,
            count_embed,
        })
    }

    pub fn forward_from_encoder_output(
        &self,
        last_hidden_state: &Tensor,
        text_word_positions: &[usize],
        schema_special_positions: &[usize],
    ) -> Tensor {
        let b = last_hidden_state.size()[0];
        assert_eq!(b, 1, "batch size 1 only");
        let seq = last_hidden_state.select(0, 0);

        let mut tw = Vec::new();
        for &p in text_word_positions {
            tw.push(seq.select(0, p as i64).unsqueeze(0));
        }
        let text_word_embs = Tensor::cat(&tw, 0);

        let mut sp = Vec::new();
        for &p in schema_special_positions {
            sp.push(seq.select(0, p as i64).unsqueeze(0));
        }
        let schema_special_embs = Tensor::cat(&sp, 0);

        let text_len = text_word_embs.size()[0] as usize;
        let mut span_data = Vec::with_capacity(text_len * self.max_width * 2);
        for i in 0..text_len {
            for w in 0..self.max_width {
                let end = (i + w).min(text_len - 1);
                span_data.push(i as i64);
                span_data.push(end as i64);
            }
        }
        let dev = last_hidden_state.device();
        let span_idx = Tensor::from_slice(&span_data)
            .to_device(dev)
            .to_kind(Kind::Int64)
            .reshape([1, (text_len * self.max_width) as i64, 2]);

        let span_rep = self
            .span_rep
            .forward(&text_word_embs.unsqueeze(0), &span_idx)
            .select(0, 0);

        let p_emb = schema_special_embs.select(0, 0);
        let count_logits = self.count_pred.forward(&p_emb.unsqueeze(0));
        let pred_count = count_logits.argmax(-1, false).int64_value(&[0]) as usize;

        let num_entities = schema_special_embs.size()[0] - 1;
        if pred_count == 0 {
            return Tensor::zeros(
                [num_entities, text_len as i64, self.max_width as i64],
                (Kind::Float, dev),
            );
        }

        let e_embs = schema_special_embs.narrow(0, 1, num_entities);
        let struct_proj = self.count_embed.forward(&e_embs, pred_count);
        let struct_proj_0 = struct_proj.select(0, 0);

        let d = span_rep.size()[2];
        let flat_span = span_rep.reshape([-1, d]);
        let scores = flat_span.matmul(&struct_proj_0.transpose(0, 1)).sigmoid();
        scores
            .transpose(0, 1)
            .reshape([num_entities, text_len as i64, self.max_width as i64])
    }

    pub fn compute_span_rep(&self, text_word_embs: &Tensor) -> Tensor {
        let text_len = text_word_embs.size()[0] as usize;
        let dev = text_word_embs.device();
        let mut span_data = Vec::with_capacity(text_len * self.max_width * 2);
        for i in 0..text_len {
            for w in 0..self.max_width {
                let end = (i + w).min(text_len - 1);
                span_data.push(i as i64);
                span_data.push(end as i64);
            }
        }
        let span_idx = Tensor::from_slice(&span_data)
            .to_device(dev)
            .to_kind(Kind::Int64)
            .reshape([1, (text_len * self.max_width) as i64, 2]);
        self.span_rep
            .forward(&text_word_embs.unsqueeze(0), &span_idx)
            .select(0, 0)
    }

    pub fn compute_span_rep_batched(&self, token_embs_list: &[Tensor]) -> Vec<Tensor> {
        if token_embs_list.is_empty() {
            return vec![];
        }
        let device = token_embs_list[0].device();
        let mut text_lengths = Vec::new();
        let mut hidden = None;
        for t in token_embs_list {
            let (l, d) = t.size2().unwrap();
            text_lengths.push(l as usize);
            match hidden {
                None => hidden = Some(d),
                Some(h) if h != d => panic!("hidden dim mismatch"),
                _ => {}
            }
        }
        let hidden = hidden.expect("hidden dim");
        let hidden_us = hidden as usize;
        let max_text_len = *text_lengths.iter().max().unwrap() as i64;
        let batch_size = token_embs_list.len() as i64;
        let max_w = self.max_width as i64;
        let n_spans = max_text_len * max_w;

        let mut padded = vec![0f32; (batch_size * max_text_len * hidden) as usize];
        for (bi, emb) in token_embs_list.iter().enumerate() {
            let bi = bi as i64;
            let li = text_lengths[bi as usize] as i64;
            let mut buf = vec![0f32; (li * hidden) as usize];
            let blen = buf.len();
            emb.copy_data(&mut buf, blen);
            for j in 0..li {
                let src = (j * hidden) as usize;
                let dst = ((bi * max_text_len + j) * hidden) as usize;
                padded[dst..dst + hidden_us].copy_from_slice(&buf[src..src + hidden_us]);
            }
        }
        let padded_t = Tensor::from_slice(&padded).to_device(device).reshape([
            batch_size,
            max_text_len,
            hidden,
        ]);

        let mut safe_flat = vec![0i64; (batch_size * n_spans * 2) as usize];
        for (b, &tl_us) in text_lengths.iter().enumerate() {
            let b = b as i64;
            let tl = tl_us as i64;
            for i in 0..max_text_len {
                for w in 0..max_w {
                    let idx = i * max_w + w;
                    let flat_base = ((b * n_spans + idx) * 2) as usize;
                    let end = i + w;
                    if end < tl {
                        safe_flat[flat_base] = i;
                        safe_flat[flat_base + 1] = end;
                    }
                }
            }
        }
        let safe_spans = Tensor::from_slice(&safe_flat)
            .to_device(device)
            .to_kind(Kind::Int64)
            .reshape([batch_size, n_spans, 2]);
        let span_rep = self.span_rep.forward(&padded_t, &safe_spans);

        let mut out = Vec::with_capacity(batch_size as usize);
        for (b, &tl) in text_lengths.iter().enumerate() {
            let row = span_rep.select(0, b as i64).narrow(0, 0, tl as i64);
            out.push(row);
        }
        out
    }

    pub fn classifier_logits(&self, label_rows: &Tensor) -> Tensor {
        self.classifier.forward(label_rows).squeeze_copy_dim(-1)
    }

    pub fn count_predict(&self, p_embedding: &Tensor) -> usize {
        let count_logits = self.count_pred.forward(&p_embedding.unsqueeze(0));
        count_logits.argmax(-1, false).int64_value(&[0]) as usize
    }

    pub fn span_scores_sigmoid(
        &self,
        span_rep: &Tensor,
        field_embs: &Tensor,
        pred_count: usize,
    ) -> Tensor {
        let l = span_rep.size()[0];
        let max_w = span_rep.size()[1];
        let d = span_rep.size()[2];
        let p = field_embs.size()[0];
        let d2 = field_embs.size()[1];
        assert_eq!(d, d2);
        let struct_proj = self.count_embed.forward(field_embs, pred_count);
        let span_flat = span_rep.reshape([l * max_w, d]);
        let mut planes = Vec::with_capacity(pred_count);
        for b in 0..pred_count as i64 {
            let sb = struct_proj.select(0, b);
            let scores = span_flat.matmul(&sb.transpose(0, 1));
            let scores = scores.transpose(0, 1).reshape([p, l, max_w]);
            planes.push(scores.unsqueeze(0));
        }
        Tensor::cat(&planes, 0).sigmoid()
    }
}
