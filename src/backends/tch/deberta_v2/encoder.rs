// Copyright 2020, Microsoft and the HuggingFace Inc. team.
// Copyright 2022 Guillaume Becquin
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Derived from rust-bert (https://github.com/guillaume-be/rust-bert)

use super::attention::{
    BaseDebertaLayerNorm, DebertaAttention, DebertaV2DisentangledSelfAttention,
    DisentangledSelfAttention, build_relative_position,
};
use super::common::{
    Activation, TensorFunction, XDropout, get_shape_and_device_from_ids_embeddings_pair,
    process_ids_embeddings_pair,
};
use super::config::{DebertaConfig, DebertaV2Config, NormRelEmbedType};
use anyhow::Result;
use std::borrow::{Borrow, BorrowMut};
use tch::nn::{ConvConfig, EmbeddingConfig, LayerNorm, LayerNormConfig, Module};
use tch::{Kind, Tensor, nn};

// ---------------------------------------------------------------------------
// Embeddings
// ---------------------------------------------------------------------------

pub struct BaseDebertaEmbeddings<LN>
where
    LN: BaseDebertaLayerNorm + Module,
{
    word_embeddings: nn::Embedding,
    position_embeddings: Option<nn::Embedding>,
    token_type_embeddings: Option<nn::Embedding>,
    embed_proj: Option<nn::Linear>,
    layer_norm: LN,
    dropout: XDropout,
}

impl<LN> BaseDebertaEmbeddings<LN>
where
    LN: BaseDebertaLayerNorm + Module,
{
    pub fn new<'p, P>(p: P, config: &DebertaConfig) -> BaseDebertaEmbeddings<LN>
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();

        let embedding_config = EmbeddingConfig {
            padding_idx: config.pad_token_id.unwrap_or(0),
            ..Default::default()
        };
        let embedding_size = config.embedding_size.unwrap_or(config.hidden_size);

        let word_embeddings = nn::embedding(
            p / "word_embeddings",
            config.vocab_size,
            embedding_size,
            embedding_config,
        );

        let position_embeddings = if config.position_biased_input.unwrap_or(true) {
            Some(nn::embedding(
                p / "position_embeddings",
                config.max_position_embeddings,
                embedding_size,
                Default::default(),
            ))
        } else {
            None
        };

        let token_type_embeddings = if config.type_vocab_size > 0 {
            Some(nn::embedding(
                p / "token_type_embeddings",
                config.type_vocab_size,
                embedding_size,
                Default::default(),
            ))
        } else {
            None
        };

        let embed_proj = if embedding_size != config.hidden_size {
            let linear_config = nn::LinearConfig {
                bias: false,
                ..Default::default()
            };
            Some(nn::linear(
                p / "embed_proj",
                embedding_size,
                config.hidden_size,
                linear_config,
            ))
        } else {
            None
        };

        let layer_norm = LN::new(
            p / "LayerNorm",
            embedding_size,
            config.layer_norm_eps.unwrap_or(1e-7),
        );
        let dropout = XDropout::new(config.hidden_dropout_prob);
        BaseDebertaEmbeddings {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            embed_proj,
            layer_norm,
            dropout,
        }
    }

    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        attention_mask: &Tensor,
        input_embeds: Option<&Tensor>,
        train: bool,
    ) -> Result<Tensor> {
        let (calc_input_embeddings, input_shape, _) =
            process_ids_embeddings_pair(input_ids, input_embeds, &self.word_embeddings)?;

        let mut input_embeddings = input_embeds
            .unwrap_or_else(|| calc_input_embeddings.as_ref().unwrap())
            .shallow_clone();
        let seq_length = input_embeddings.size()[1];

        let calc_position_ids = if position_ids.is_none() {
            Some(
                Tensor::arange(seq_length, (Kind::Int64, input_embeddings.device()))
                    .expand([1, -1], true),
            )
        } else {
            None
        };

        let calc_token_type_ids = if token_type_ids.is_none() {
            Some(Tensor::zeros(
                input_shape,
                (Kind::Int64, input_embeddings.device()),
            ))
        } else {
            None
        };

        let position_ids = position_ids.unwrap_or_else(|| calc_position_ids.as_ref().unwrap());
        let token_type_ids =
            token_type_ids.unwrap_or_else(|| calc_token_type_ids.as_ref().unwrap());

        if let Some(position_embeddings) = &self.position_embeddings {
            let position_embeddings = position_ids.apply(position_embeddings);
            input_embeddings = input_embeddings + position_embeddings;
        };

        if let Some(token_type_embeddings) = &self.token_type_embeddings {
            let token_type_embeddings = token_type_ids.apply(token_type_embeddings);
            input_embeddings = input_embeddings + token_type_embeddings;
        };

        if let Some(embed_proj) = &self.embed_proj {
            input_embeddings = input_embeddings.apply(embed_proj);
        };

        input_embeddings = input_embeddings.apply(&self.layer_norm);

        let mask = if attention_mask.dim() != input_embeddings.dim() {
            if attention_mask.dim() != 4 {
                attention_mask
                    .squeeze_dim(1)
                    .squeeze_dim(1)
                    .unsqueeze(2)
                    .to_kind(input_embeddings.kind())
            } else {
                attention_mask.unsqueeze(2).to_kind(input_embeddings.kind())
            }
        } else {
            attention_mask.to_kind(input_embeddings.kind())
        };
        input_embeddings = input_embeddings * mask;

        Ok(input_embeddings.apply_t(&self.dropout, train))
    }
}

pub type DebertaV2Embeddings = BaseDebertaEmbeddings<LayerNorm>;

// ---------------------------------------------------------------------------
// Intermediate + Output layers
// ---------------------------------------------------------------------------

pub struct DebertaIntermediate {
    dense: nn::Linear,
    activation: TensorFunction,
}

impl DebertaIntermediate {
    pub fn new<'p, P>(p: P, config: &DebertaConfig) -> DebertaIntermediate
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        let dense = nn::linear(
            p / "dense",
            config.hidden_size,
            config.intermediate_size,
            Default::default(),
        );
        let activation = config.hidden_act.get_function();
        DebertaIntermediate { dense, activation }
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Tensor {
        (self.activation.get_fn())(&hidden_states.apply(&self.dense))
    }
}

pub struct DebertaOutput<LN: BaseDebertaLayerNorm + Module> {
    dense: nn::Linear,
    layer_norm: LN,
    dropout: XDropout,
}

impl<LN: BaseDebertaLayerNorm + Module> DebertaOutput<LN> {
    pub fn new<'p, P>(p: P, config: &DebertaConfig) -> DebertaOutput<LN>
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        let dense = nn::linear(
            p / "dense",
            config.intermediate_size,
            config.hidden_size,
            Default::default(),
        );
        let layer_norm = LN::new(
            p / "LayerNorm",
            config.hidden_size,
            config.layer_norm_eps.unwrap_or(1e-7),
        );
        let dropout = XDropout::new(config.hidden_dropout_prob);
        DebertaOutput {
            dense,
            layer_norm,
            dropout,
        }
    }

    pub fn forward_t(&self, hidden_states: &Tensor, input_tensor: &Tensor, train: bool) -> Tensor {
        let hidden_states: Tensor = input_tensor
            + hidden_states
                .apply(&self.dense)
                .apply_t(&self.dropout, train);
        hidden_states.apply(&self.layer_norm)
    }
}

// ---------------------------------------------------------------------------
// BaseDebertaLayer (generic transformer layer)
// ---------------------------------------------------------------------------

pub struct BaseDebertaLayer<SA, LN>
where
    SA: DisentangledSelfAttention,
    LN: BaseDebertaLayerNorm + Module,
{
    attention: DebertaAttention<SA, LN>,
    intermediate: DebertaIntermediate,
    output: DebertaOutput<LN>,
}

impl<SA, LN> BaseDebertaLayer<SA, LN>
where
    SA: DisentangledSelfAttention,
    LN: BaseDebertaLayerNorm + Module,
{
    pub fn new<'p, P>(p: P, config: &DebertaConfig) -> BaseDebertaLayer<SA, LN>
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        let attention = DebertaAttention::new(p / "attention", config);
        let intermediate = DebertaIntermediate::new(p / "intermediate", config);
        let output = DebertaOutput::new(p / "output", config);

        BaseDebertaLayer {
            attention,
            intermediate,
            output,
        }
    }

    pub fn forward_t(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
        query_states: Option<&Tensor>,
        relative_pos: Option<&Tensor>,
        relative_embeddings: Option<&Tensor>,
        train: bool,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let (attention_output, attention_matrix) = self.attention.forward_t(
            hidden_states,
            attention_mask,
            query_states,
            relative_pos,
            relative_embeddings,
            train,
        )?;

        let intermediate_output = self.intermediate.forward(&attention_output);
        let layer_output = self
            .output
            .forward_t(&intermediate_output, &attention_output, train);

        Ok((layer_output, attention_matrix))
    }
}

pub type DebertaV2Layer = BaseDebertaLayer<DebertaV2DisentangledSelfAttention, LayerNorm>;

// ---------------------------------------------------------------------------
// Encoder output
// ---------------------------------------------------------------------------

pub struct DebertaEncoderOutput {
    pub hidden_state: Tensor,
    pub all_hidden_states: Option<Vec<Tensor>>,
    pub all_attentions: Option<Vec<Tensor>>,
}

// ---------------------------------------------------------------------------
// ConvLayer
// ---------------------------------------------------------------------------

pub struct ConvLayer {
    conv: nn::Conv1D,
    layer_norm: nn::LayerNorm,
    dropout: XDropout,
    conv_act: TensorFunction,
}

impl ConvLayer {
    pub fn new<'p, P>(p: P, config: &DebertaV2Config) -> ConvLayer
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        let conv_act = config.conv_act.unwrap_or(Activation::tanh).get_function();
        let kernel_size = config.conv_kernel_size.unwrap_or(3);
        let groups = config.conv_groups.unwrap_or(1);

        let conv_config = ConvConfig {
            padding: (kernel_size - 1) / 2,
            groups,
            ..Default::default()
        };
        let conv = nn::conv1d(
            p / "conv",
            config.hidden_size,
            config.hidden_size,
            kernel_size,
            conv_config,
        );

        let layer_norm_config = LayerNormConfig {
            eps: config.layer_norm_eps.unwrap_or(1e-7),
            ..Default::default()
        };
        let layer_norm =
            nn::layer_norm(p / "LayerNorm", vec![config.hidden_size], layer_norm_config);

        let dropout = XDropout::new(config.hidden_dropout_prob);

        ConvLayer {
            conv,
            layer_norm,
            dropout,
            conv_act,
        }
    }

    pub fn forward_t(
        &self,
        hidden_states: &Tensor,
        residual_states: &Tensor,
        input_mask: &Tensor,
        train: bool,
    ) -> Tensor {
        let out = hidden_states
            .permute([0, 2, 1])
            .contiguous()
            .apply(&self.conv)
            .permute([0, 2, 1])
            .contiguous();
        let reverse_mask: Tensor = 1 - input_mask;
        let out = out.masked_fill(
            &reverse_mask
                .to_kind(Kind::Bool)
                .unsqueeze(-1)
                .expand(out.size().as_slice(), true),
            0,
        );
        let out = self.conv_act.get_fn()(&out.apply_t(&self.dropout, train));

        let layer_norm_input = residual_states + out;
        let output = layer_norm_input.apply(&self.layer_norm);
        let new_input_mask = if input_mask.dim() != layer_norm_input.dim() {
            if input_mask.dim() == 4 {
                input_mask.squeeze_dim(1).squeeze_dim(1).unsqueeze(2)
            } else {
                input_mask.unsqueeze(2)
            }
            .to_kind(output.kind())
        } else {
            input_mask.to_kind(output.kind())
        };
        output * new_input_mask
    }
}

// ---------------------------------------------------------------------------
// DebertaV2Encoder
// ---------------------------------------------------------------------------

pub struct DebertaV2Encoder {
    output_attentions: bool,
    output_hidden_states: bool,
    layers: Vec<DebertaV2Layer>,
    max_relative_positions: Option<i64>,
    position_buckets: Option<i64>,
    rel_embeddings: Option<nn::Embedding>,
    layer_norm: Option<nn::LayerNorm>,
    conv: Option<ConvLayer>,
}

impl DebertaV2Encoder {
    pub fn new<'p, P>(p: P, config: &DebertaV2Config) -> DebertaV2Encoder
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        let output_attentions = config.output_attentions.unwrap_or(false);
        let output_hidden_states = config.output_hidden_states.unwrap_or(false);
        let p_layer = p / "layer";
        let mut layers: Vec<DebertaV2Layer> = vec![];
        for layer_index in 0..config.num_hidden_layers {
            layers.push(DebertaV2Layer::new(
                &p_layer / layer_index,
                &(config.into()),
            ));
        }
        let (rel_embeddings, max_relative_positions, position_buckets) =
            if config.relative_attention.unwrap_or(false) {
                let mut max_relative_positions = config.max_relative_positions.unwrap_or(-1);
                if max_relative_positions < 1 {
                    max_relative_positions = config.max_position_embeddings;
                };
                let position_buckets = config.position_buckets.unwrap_or(-1);
                let position_embed_size = if position_buckets > 0 {
                    position_buckets * 2
                } else {
                    max_relative_positions * 2
                };
                let rel_embeddings = nn::embedding(
                    p / "rel_embeddings",
                    position_embed_size,
                    config.hidden_size,
                    Default::default(),
                );
                (
                    Some(rel_embeddings),
                    Some(max_relative_positions),
                    Some(position_buckets),
                )
            } else {
                (None, None, None)
            };

        let layer_norm = if config
            .norm_rel_ebd
            .clone()
            .unwrap_or_default()
            .has_type(NormRelEmbedType::layer_norm)
        {
            Some(nn::layer_norm(
                p / "LayerNorm",
                vec![config.hidden_size],
                LayerNormConfig {
                    eps: 1e-7,
                    elementwise_affine: true,
                    ..Default::default()
                },
            ))
        } else {
            None
        };

        let conv = if config.conv_kernel_size.unwrap_or(0) > 0 {
            Some(ConvLayer::new(p / "conv", config))
        } else {
            None
        };

        DebertaV2Encoder {
            output_attentions,
            output_hidden_states,
            layers,
            max_relative_positions,
            position_buckets,
            rel_embeddings,
            layer_norm,
            conv,
        }
    }

    fn get_rel_embedding(&self) -> Option<Tensor> {
        self.rel_embeddings.as_ref().map(|embeddings| {
            let rel_embeds = &embeddings.ws;
            if let Some(layer_norm) = &self.layer_norm {
                rel_embeds.apply(layer_norm)
            } else {
                rel_embeds.shallow_clone()
            }
        })
    }

    /// Match candle-transformers `DebertaV2Encoder::get_attention_mask` (HF DeBERTa V2):
    /// broadcast `[B, L]` to a pairwise `[B, 1, L, L]` content mask (float), no uint8 conversion.
    fn get_attention_mask(attention_mask: &Tensor) -> Tensor {
        match attention_mask.dim() {
            d if d <= 2 => {
                let m = attention_mask.to_kind(Kind::Float);
                let extended_attention_mask = m.unsqueeze(1).unsqueeze(2);
                let right = extended_attention_mask.squeeze_dim(-2).unsqueeze(-1);
                extended_attention_mask * right
            }
            3 => attention_mask.unsqueeze(1),
            _ => attention_mask.shallow_clone(),
        }
    }

    fn reverse_vec<T>(mut input_vec: Vec<T>) -> Vec<T> {
        input_vec.reverse();
        input_vec
    }

    fn get_rel_pos(
        &self,
        hidden_states: &Tensor,
        query_states: Option<&Tensor>,
        relative_pos: Option<&Tensor>,
    ) -> Option<Tensor> {
        if self.rel_embeddings.is_some() & relative_pos.is_none() {
            let q = query_states
                .map(|query_states| DebertaV2Encoder::reverse_vec(query_states.size())[1])
                .unwrap_or_else(|| DebertaV2Encoder::reverse_vec(hidden_states.size())[1]);

            Some(build_relative_position(
                q,
                DebertaV2Encoder::reverse_vec(hidden_states.size())[1],
                self.position_buckets.unwrap(),
                self.max_relative_positions.unwrap(),
                hidden_states.device(),
            ))
        } else {
            relative_pos.map(|tensor| tensor.shallow_clone())
        }
    }

    pub fn forward_t(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
        query_states: Option<&Tensor>,
        relative_pos: Option<&Tensor>,
        train: bool,
    ) -> Result<DebertaEncoderOutput> {
        let mut all_hidden_states: Option<Vec<Tensor>> = if self.output_hidden_states {
            Some(vec![])
        } else {
            None
        };
        let mut all_attentions: Option<Vec<Tensor>> = if self.output_attentions {
            Some(vec![])
        } else {
            None
        };

        let input_mask = if attention_mask.dim() <= 2 {
            attention_mask.shallow_clone()
        } else {
            attention_mask
                .sum_dim_intlist([-2].as_slice(), false, attention_mask.kind())
                .gt(0)
                .to_kind(Kind::Uint8)
        };
        let attention_mask = Self::get_attention_mask(attention_mask);
        let relative_pos = self.get_rel_pos(hidden_states, query_states, relative_pos);
        let relative_embeddings = self.get_rel_embedding();

        let mut output_states = None::<Tensor>;
        let mut attention_weights: Option<Tensor>;

        for (layer_index, layer) in self.layers.iter().enumerate() {
            let layer_output = if let Some(output_states) = &output_states {
                layer.forward_t(
                    output_states,
                    &attention_mask,
                    query_states,
                    relative_pos.as_ref(),
                    relative_embeddings.as_ref(),
                    train,
                )?
            } else {
                layer.forward_t(
                    hidden_states,
                    &attention_mask,
                    query_states,
                    relative_pos.as_ref(),
                    relative_embeddings.as_ref(),
                    train,
                )?
            };

            output_states = Some(layer_output.0);
            if layer_index == 0 {
                if let Some(conv) = &self.conv {
                    output_states = output_states.map(|output_states| {
                        conv.forward_t(hidden_states, &output_states, &input_mask, train)
                    })
                }
            }
            attention_weights = layer_output.1;
            if let Some(attentions) = all_attentions.borrow_mut() {
                attentions.push(std::mem::take(&mut attention_weights.unwrap()));
            };
            if let Some(hidden_states) = all_hidden_states.borrow_mut() {
                hidden_states.push(output_states.as_ref().unwrap().copy());
            };
        }

        Ok(DebertaEncoderOutput {
            hidden_state: output_states.unwrap(),
            all_hidden_states,
            all_attentions,
        })
    }
}

// ---------------------------------------------------------------------------
// DebertaV2Model
// ---------------------------------------------------------------------------

pub struct DebertaV2Model {
    embeddings: DebertaV2Embeddings,
    encoder: DebertaV2Encoder,
}

impl DebertaV2Model {
    pub fn new<'p, P>(p: P, config: &DebertaV2Config) -> DebertaV2Model
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        let embeddings = DebertaV2Embeddings::new(p / "embeddings", &config.into());
        let encoder = DebertaV2Encoder::new(p / "encoder", config);
        DebertaV2Model {
            embeddings,
            encoder,
        }
    }

    pub fn forward_t(
        &self,
        input_ids: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        input_embeds: Option<&Tensor>,
        train: bool,
    ) -> Result<DebertaEncoderOutput> {
        let (input_shape, device) =
            get_shape_and_device_from_ids_embeddings_pair(input_ids, input_embeds)?;

        let calc_attention_mask = if attention_mask.is_none() {
            Some(Tensor::ones(input_shape.as_slice(), (Kind::Bool, device)))
        } else {
            None
        };

        let attention_mask =
            attention_mask.unwrap_or_else(|| calc_attention_mask.as_ref().unwrap());

        let embedding_output = self.embeddings.forward_t(
            input_ids,
            token_type_ids,
            position_ids,
            attention_mask,
            input_embeds,
            train,
        )?;

        let encoder_output =
            self.encoder
                .forward_t(&embedding_output, attention_mask, None, None, train)?;

        Ok(encoder_output)
    }
}
