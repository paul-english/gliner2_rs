use super::deberta_v2::{DebertaV2Config, DebertaV2Model};
use super::layers::{CountEmbed, Mlp2};
use super::span_rep::SpanMarkerV0;
use super::tensor_wrap::BurnTensor;
use super::weights::WeightMap;
use crate::config::{ExtractorConfig, ModelFiles};
use anyhow::{Context, Result};
use burn::backend::NdArray;
use burn::prelude::*;
use burn::tensor::activation;

type B = NdArray;

pub struct BurnExtractor {
    encoder: DebertaV2Model<B>,
    span_rep: SpanMarkerV0<B>,
    classifier: Mlp2<B>,
    count_pred: Mlp2<B>,
    count_embed: CountEmbed<B>,
    hidden_size: usize,
    max_width: usize,
    device: <B as Backend>::Device,
}

// NdArray tensors are backed by ndarray::Array (Arc storage), which is Send+Sync.
// No mutation after construction.
unsafe impl Send for BurnExtractor {}
unsafe impl Sync for BurnExtractor {}

impl BurnExtractor {
    pub fn load_cpu(
        files: &ModelFiles,
        config: ExtractorConfig,
        vocab_size: usize,
    ) -> Result<Self> {
        let device = Default::default();
        Self::load(files, config, vocab_size, &device)
    }

    pub fn load(
        files: &ModelFiles,
        config: ExtractorConfig,
        vocab_size: usize,
        device: &<B as Backend>::Device,
    ) -> Result<Self> {
        let mut encoder_config: DebertaV2Config =
            serde_json::from_str(&std::fs::read_to_string(&files.encoder_config)?)
                .context("parse encoder_config")?;
        encoder_config.vocab_size = vocab_size;

        let map = WeightMap::from_safetensors(&files.weights)?;
        let hidden_size = encoder_config.hidden_size;
        let max_width = config.max_width;

        let encoder = DebertaV2Model::load(&map, "encoder", &encoder_config, device)?;
        let span_rep = SpanMarkerV0::load(
            &map,
            "span_rep.span_rep_layer",
            hidden_size,
            max_width,
            device,
        )?;
        let classifier = Mlp2::load(&map, "classifier", device)?;
        let count_pred = Mlp2::load(&map, "count_pred", device)?;
        let count_embed = CountEmbed::load(
            &config.counting_layer,
            hidden_size,
            &map,
            "count_embed",
            device,
        )?;

        Ok(Self {
            encoder,
            span_rep,
            classifier,
            count_pred,
            count_embed,
            hidden_size,
            max_width,
            device: device.clone(),
        })
    }

    // --- internal helpers ---

    fn gather_text_word_embeddings_row(
        &self,
        last_hidden_seq: &Tensor<B, 2>,
        positions: &[usize],
    ) -> Tensor<B, 2> {
        let rows: Vec<Tensor<B, 2>> = positions
            .iter()
            .map(|&p| {
                last_hidden_seq
                    .clone()
                    .narrow(0, p, 1) // [1, D]
            })
            .collect();
        Tensor::cat(rows, 0) // [L, D]
    }
}

impl crate::engine::Gliner2Engine for BurnExtractor {
    type Tensor = BurnTensor<B>;

    fn dup_tensor(&self, t: &Self::Tensor) -> Self::Tensor {
        t.clone()
    }

    fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    fn max_width(&self) -> usize {
        self.max_width
    }

    fn encode_sequence(
        &self,
        input_ids: &Self::Tensor,
        attention_mask: &Self::Tensor,
    ) -> Result<Self::Tensor> {
        let ids = input_ids.as_i2()?;
        let mask = attention_mask.as_i2()?;
        let hidden = self.encoder.forward(ids, mask);
        Ok(BurnTensor::F3(hidden))
    }

    fn gather_text_word_embeddings(
        &self,
        last_hidden: &Self::Tensor,
        positions: &[usize],
    ) -> Result<Self::Tensor> {
        let hidden = last_hidden.as_f3()?;
        let row: Tensor<B, 2> = hidden.clone().narrow(0, 0, 1).squeeze_dim::<2>(0); // [S, D]
        Ok(BurnTensor::F2(
            self.gather_text_word_embeddings_row(&row, positions),
        ))
    }

    fn gather_text_word_embeddings_batch_idx(
        &self,
        last_hidden: &Self::Tensor,
        batch_idx: usize,
        positions: &[usize],
    ) -> Result<Self::Tensor> {
        let hidden = last_hidden.as_f3()?;
        let row: Tensor<B, 2> = hidden.clone().narrow(0, batch_idx, 1).squeeze_dim::<2>(0);
        Ok(BurnTensor::F2(
            self.gather_text_word_embeddings_row(&row, positions),
        ))
    }

    fn compute_span_rep(&self, text_word_embs: &Self::Tensor) -> Result<Self::Tensor> {
        let embs = text_word_embs.as_f2()?; // [L, D]
        let text_len = embs.dims()[0];
        let indices = crate::span_utils::generate_span_indices(text_len, self.max_width);
        let span_flat: Vec<i64> = indices
            .iter()
            .flat_map(|[s, e]| [*s as i64, *e as i64])
            .collect();
        let n_spans = indices.len();
        let span_indices: Tensor<B, 3, Int> =
            Tensor::from_data(
                burn::tensor::TensorData::new(span_flat, [1, n_spans, 2]),
                &self.device,
            );

        let h: Tensor<B, 3> = embs.clone().unsqueeze_dim::<3>(0); // [1, L, D]
        let span_rep = self.span_rep.forward(&h, &span_indices); // [1, L, max_width, D]
        let [_, l, k, d] = span_rep.dims();
        let span_rep_3d: Tensor<B, 3> = span_rep.reshape([l, k, d]);
        Ok(BurnTensor::F3(span_rep_3d))
    }

    fn compute_span_rep_batched(
        &self,
        token_embs_list: &[Self::Tensor],
    ) -> Result<Vec<Self::Tensor>> {
        if token_embs_list.is_empty() {
            return Ok(vec![]);
        }

        let emb_list: Vec<&Tensor<B, 2>> = token_embs_list
            .iter()
            .map(|t| t.as_f2())
            .collect::<Result<_>>()?;

        let mut text_lengths = Vec::with_capacity(emb_list.len());
        let mut hidden = 0usize;
        for e in &emb_list {
            let [l, d] = e.dims();
            text_lengths.push(l);
            hidden = d;
        }
        let max_text_len = *text_lengths.iter().max().unwrap();
        let batch_size = emb_list.len();
        let max_width = self.max_width;
        let n_spans = max_text_len * max_width;

        // Pad all embeddings to max_text_len
        let mut padded_rows = Vec::with_capacity(batch_size);
        for e in &emb_list {
            let [l, _d] = e.dims();
            if l < max_text_len {
                let pad: Tensor<B, 2> = Tensor::zeros([max_text_len - l, hidden], &self.device);
                padded_rows.push(Tensor::cat(vec![(*e).clone(), pad], 0));
            } else {
                padded_rows.push((*e).clone());
            }
        }
        let padded_rows: Vec<Tensor<B, 3>> =
            padded_rows.into_iter().map(|t| t.unsqueeze_dim::<3>(0)).collect();
        let padded_t: Tensor<B, 3> = Tensor::cat(padded_rows, 0); // [B, max_L, D]

        let batched_indices = crate::span_utils::generate_batched_span_indices(
            &text_lengths,
            max_text_len,
            max_width,
        );
        let safe_flat: Vec<i64> = batched_indices
            .iter()
            .flat_map(|[s, e]| [*s as i64, *e as i64])
            .collect();
        let safe_spans: Tensor<B, 3, Int> = Tensor::from_data(
            burn::tensor::TensorData::new(safe_flat, [batch_size, n_spans, 2]),
            &self.device,
        );

        let span_rep = self.span_rep.forward(&padded_t, &safe_spans);
        // span_rep: [B, max_L, max_width, D]
        let [_b, _ml, _mw, d] = span_rep.dims();

        let mut out = Vec::with_capacity(batch_size);
        for (bi, &tl) in text_lengths.iter().enumerate() {
            let row: Tensor<B, 3> = span_rep.clone().narrow(0, bi, 1).squeeze_dim::<3>(0); // [max_L, max_width, D]
            let row = row.narrow(0, 0, tl); // [tl, max_width, D]
            out.push(BurnTensor::F3(row));
        }
        Ok(out)
    }

    fn classifier_logits(&self, label_rows: &Self::Tensor) -> Result<Self::Tensor> {
        let rows = label_rows.as_f2()?; // [N, D]
        let logits = self.classifier.forward(rows); // [N, 1]
        let n = logits.dims()[0];
        Ok(BurnTensor::F1(logits.reshape([n])))
    }

    fn count_predict(&self, p_embedding: &Self::Tensor) -> Result<usize> {
        let emb = p_embedding.as_f1()?; // [D]
        let d = emb.dims()[0];
        let input = emb.clone().reshape([1, d]);
        let logits = self.count_pred.forward(&input); // [1, 20]
        let pred = logits.argmax(1); // [1, 1] int
        let data = pred.to_data();
        let val: i32 = data.to_vec::<i32>().unwrap()[0];
        Ok(val as usize)
    }

    fn span_scores_sigmoid(
        &self,
        span_rep: &Self::Tensor,
        field_embs: &Self::Tensor,
        pred_count: usize,
    ) -> Result<Self::Tensor> {
        let span = span_rep.as_f3()?; // [L, K, D]
        let fields = field_embs.as_f2()?; // [P, D]
        let [l, max_w, d] = span.dims();
        let p = fields.dims()[0];

        let struct_proj = self.count_embed.forward(fields, pred_count); // [count, P, D]
        let span_flat = span.clone().reshape([l * max_w, d]);

        let mut planes = Vec::with_capacity(pred_count);
        for b in 0..pred_count {
            let sb: Tensor<B, 2> = struct_proj.clone().narrow(0, b, 1).squeeze_dim::<2>(0); // [P, D]
            let scores = span_flat.clone().matmul(sb.transpose()); // [L*K, P]
            let scores = scores.transpose().reshape([p, l, max_w]); // [P, L, K]
            planes.push(scores.unsqueeze_dim::<4>(0));
        }
        let stacked = Tensor::cat(planes, 0); // [count, P, L, K]
        Ok(BurnTensor::F4(activation::sigmoid(stacked)))
    }

    fn single_sample_inputs(&self, input_ids: &[u32]) -> Result<(Self::Tensor, Self::Tensor)> {
        let s = input_ids.len();
        let ids_i64: Vec<i64> = input_ids.iter().map(|&x| x as i64).collect();
        let ids: Tensor<B, 2, Int> = Tensor::from_data(
            burn::tensor::TensorData::new(ids_i64, [1, s]),
            &self.device,
        );
        let mask: Tensor<B, 2, Int> = Tensor::ones([1, s], &self.device);
        Ok((BurnTensor::I2(ids), BurnTensor::I2(mask)))
    }

    fn batch_inputs(
        &self,
        input_ids: Vec<u32>,
        attention_mask_i64: Vec<i64>,
        batch_size: usize,
        max_seq_len: usize,
    ) -> Result<(Self::Tensor, Self::Tensor)> {
        let ids_i64: Vec<i64> = input_ids.iter().map(|&x| x as i64).collect();
        let ids: Tensor<B, 2, Int> = Tensor::from_data(
            burn::tensor::TensorData::new(ids_i64, [batch_size, max_seq_len]),
            &self.device,
        );
        let mask: Tensor<B, 2, Int> = Tensor::from_data(
            burn::tensor::TensorData::new(attention_mask_i64, [batch_size, max_seq_len]),
            &self.device,
        );
        Ok((BurnTensor::I2(ids), BurnTensor::I2(mask)))
    }

    fn batch_row_hidden(&self, hidden: &Self::Tensor, idx: usize) -> Result<Self::Tensor> {
        let h = hidden.as_f3()?;
        Ok(BurnTensor::F2(h.clone().narrow(0, idx, 1).squeeze_dim::<2>(0)))
    }

    fn stack_schema_token_embeddings(
        &self,
        last_hidden_seq: &Self::Tensor,
        positions: &[usize],
    ) -> Result<Self::Tensor> {
        let seq = last_hidden_seq.as_f2()?; // [S, D]
        let rows: Vec<Tensor<B, 2>> = positions
            .iter()
            .map(|&p| seq.clone().narrow(0, p, 1))
            .collect();
        Ok(BurnTensor::F2(Tensor::cat(rows, 0)))
    }

    fn tensor_dim0(&self, t: &Self::Tensor) -> Result<usize> {
        Ok(t.dim0())
    }

    fn tensor_narrow0(&self, t: &Self::Tensor, start: usize, len: usize) -> Result<Self::Tensor> {
        t.narrow0(start, len)
    }

    fn tensor_index0(&self, t: &Self::Tensor, i: usize) -> Result<Self::Tensor> {
        t.index0(i)
    }

    fn tensor_logits_1d(&self, logits: &Self::Tensor) -> Result<Vec<f32>> {
        let t = logits.as_f1()?;
        let data = t.to_data();
        Ok(data.to_vec().unwrap())
    }

    fn tensor_span_scores_to_vec4(
        &self,
        t: &Self::Tensor,
    ) -> Result<Vec<Vec<Vec<Vec<f32>>>>> {
        let t = t.as_f4()?;
        let [a, b, c, d] = t.dims();
        let flat: Vec<f32> = t.to_data().to_vec().unwrap();
        let mut out = vec![vec![vec![vec![0f32; d]; c]; b]; a];
        let mut idx = 0;
        for i in 0..a {
            for j in 0..b {
                for k in 0..c {
                    for l in 0..d {
                        out[i][j][k][l] = flat[idx];
                        idx += 1;
                    }
                }
            }
        }
        Ok(out)
    }
}
