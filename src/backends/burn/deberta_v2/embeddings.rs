use super::config::DebertaV2Config;
use crate::backends::burn::layers::{EmbeddingW, LayerNormW, LinearNoBias};
use crate::backends::burn::weights::WeightMap;
use anyhow::Result;
use burn::prelude::*;

pub struct DebertaV2Embeddings<B: Backend> {
    word_embeddings: EmbeddingW<B>,
    position_embeddings: Option<EmbeddingW<B>>,
    token_type_embeddings: Option<EmbeddingW<B>>,
    embed_proj: Option<LinearNoBias<B>>,
    layer_norm: LayerNormW<B>,
}

impl<B: Backend> DebertaV2Embeddings<B> {
    pub fn load(
        map: &WeightMap,
        prefix: &str,
        config: &DebertaV2Config,
        device: &B::Device,
    ) -> Result<Self> {
        let word_embeddings =
            EmbeddingW::load(map, &format!("{prefix}.word_embeddings"), device)?;

        let position_embeddings = if config.position_biased_input() {
            Some(EmbeddingW::load(
                map,
                &format!("{prefix}.position_embeddings"),
                device,
            )?)
        } else {
            None
        };

        let token_type_embeddings = if config.type_vocab_size > 0 {
            Some(EmbeddingW::load(
                map,
                &format!("{prefix}.token_type_embeddings"),
                device,
            )?)
        } else {
            None
        };

        let embed_proj = if config.embedding_size() != config.hidden_size {
            Some(LinearNoBias::load(
                map,
                &format!("{prefix}.embed_proj"),
                device,
            )?)
        } else {
            None
        };

        let layer_norm = LayerNormW::load(
            map,
            &format!("{prefix}.LayerNorm"),
            config.layer_norm_eps(),
            device,
        )?;

        Ok(Self {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            embed_proj,
            layer_norm,
        })
    }

    /// `input_ids`: `[B, S]` int, `attention_mask`: `[B, S]` int → `[B, S, D]` float.
    pub fn forward(
        &self,
        input_ids: &Tensor<B, 2, Int>,
        attention_mask: &Tensor<B, 2, Int>,
    ) -> Tensor<B, 3> {
        let [b, s] = input_ids.dims();
        let device = input_ids.device();

        let mut embeddings = self.word_embeddings.forward_2d(input_ids); // [B, S, emb]

        if let Some(pos_emb) = &self.position_embeddings {
            let pos_ids: Tensor<B, 1, Int> = Tensor::arange(0..s as i64, &device);
            let pos_ids: Tensor<B, 2, Int> = pos_ids.unsqueeze_dim::<2>(0).repeat_dim(0, b); // [B, S]
            embeddings = embeddings + pos_emb.forward_2d(&pos_ids);
        }

        if let Some(tt_emb) = &self.token_type_embeddings {
            let tt_ids: Tensor<B, 2, Int> = Tensor::zeros([b, s], &device);
            embeddings = embeddings + tt_emb.forward_2d(&tt_ids);
        }

        if let Some(proj) = &self.embed_proj {
            embeddings = proj.forward_3d(&embeddings);
        }

        embeddings = self.layer_norm.forward_3d(&embeddings);

        // Mask embeddings: multiply by attention_mask (broadcast)
        let mask_f: Tensor<B, 3> = attention_mask.clone().float().unsqueeze_dim::<3>(2); // [B, S, 1]
        embeddings * mask_f
    }
}
