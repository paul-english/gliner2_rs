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

use super::common::Activation;
use serde::de::{SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, de};
use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;

// ---------------------------------------------------------------------------
// Position attention types (shared by deberta v1/v2)
// ---------------------------------------------------------------------------

#[allow(non_camel_case_types)]
#[derive(Clone, Debug, Serialize, Deserialize, Copy, PartialEq, Eq)]
pub enum PositionAttentionType {
    p2c,
    c2p,
    p2p,
}

impl FromStr for PositionAttentionType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "p2c" => Ok(PositionAttentionType::p2c),
            "c2p" => Ok(PositionAttentionType::c2p),
            "p2p" => Ok(PositionAttentionType::p2p),
            _ => Err(format!(
                "Position attention type `{s}` not in accepted variants (`p2c`, `c2p`, `p2p`)",
            )),
        }
    }
}

#[allow(non_camel_case_types)]
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct PositionAttentionTypes {
    types: Vec<PositionAttentionType>,
}

impl FromStr for PositionAttentionTypes {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let types = s
            .to_lowercase()
            .split('|')
            .map(PositionAttentionType::from_str)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(PositionAttentionTypes { types })
    }
}

impl PositionAttentionTypes {
    pub fn has_type(&self, attention_type: PositionAttentionType) -> bool {
        self.types
            .iter()
            .any(|self_type| *self_type == attention_type)
    }

    pub fn len(&self) -> usize {
        self.types.len()
    }
}

pub fn deserialize_attention_type<'de, D>(
    deserializer: D,
) -> Result<Option<PositionAttentionTypes>, D::Error>
where
    D: Deserializer<'de>,
{
    struct AttentionTypeVisitor;

    impl<'de> Visitor<'de> for AttentionTypeVisitor {
        type Value = PositionAttentionTypes;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("null, string or sequence")
        }

        fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(FromStr::from_str(value).unwrap())
        }

        fn visit_seq<S>(self, mut seq: S) -> Result<Self::Value, S::Error>
        where
            S: SeqAccess<'de>,
        {
            let mut types = vec![];
            while let Some(attention_type) = seq.next_element::<String>()? {
                types.push(FromStr::from_str(attention_type.as_str()).unwrap())
            }
            Ok(PositionAttentionTypes { types })
        }
    }

    deserializer.deserialize_any(AttentionTypeVisitor).map(Some)
}

// ---------------------------------------------------------------------------
// Norm-rel-embed types (DeBERTa v2 specific)
// ---------------------------------------------------------------------------

#[allow(non_camel_case_types)]
#[derive(Clone, Debug, Serialize, Deserialize, Copy, PartialEq, Eq)]
pub enum NormRelEmbedType {
    layer_norm,
}

impl FromStr for NormRelEmbedType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "layer_norm" => Ok(NormRelEmbedType::layer_norm),
            _ => Err(format!(
                "Layer normalization type `{s}` not in accepted variants (`layer_norm`)",
            )),
        }
    }
}

#[allow(non_camel_case_types)]
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct NormRelEmbedTypes {
    types: Vec<NormRelEmbedType>,
}

impl FromStr for NormRelEmbedTypes {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let types = s
            .to_lowercase()
            .split('|')
            .map(NormRelEmbedType::from_str)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(NormRelEmbedTypes { types })
    }
}

impl NormRelEmbedTypes {
    pub fn has_type(&self, norm_type: NormRelEmbedType) -> bool {
        self.types.iter().any(|self_type| *self_type == norm_type)
    }

    pub fn len(&self) -> usize {
        self.types.len()
    }
}

pub fn deserialize_norm_type<'de, D>(deserializer: D) -> Result<Option<NormRelEmbedTypes>, D::Error>
where
    D: Deserializer<'de>,
{
    struct NormTypeVisitor;

    impl<'de> Visitor<'de> for NormTypeVisitor {
        type Value = NormRelEmbedTypes;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("null, string or sequence")
        }

        fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(FromStr::from_str(value).unwrap())
        }

        fn visit_seq<S>(self, mut seq: S) -> Result<Self::Value, S::Error>
        where
            S: SeqAccess<'de>,
        {
            let mut types = vec![];
            while let Some(norm_type) = seq.next_element::<String>()? {
                types.push(FromStr::from_str(norm_type.as_str()).unwrap())
            }
            Ok(NormRelEmbedTypes { types })
        }
    }

    deserializer.deserialize_any(NormTypeVisitor).map(Some)
}

// ---------------------------------------------------------------------------
// DeBERTa V2 Config (public, deserialized from JSON)
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize)]
pub struct DebertaV2Config {
    pub vocab_size: i64,
    pub hidden_size: i64,
    pub num_hidden_layers: i64,
    pub hidden_act: Activation,
    pub attention_probs_dropout_prob: f64,
    pub hidden_dropout_prob: f64,
    pub initializer_range: f64,
    pub intermediate_size: i64,
    pub max_position_embeddings: i64,
    pub position_buckets: Option<i64>,
    pub num_attention_heads: i64,
    pub type_vocab_size: i64,
    pub position_biased_input: Option<bool>,
    #[serde(default, deserialize_with = "deserialize_attention_type")]
    pub pos_att_type: Option<PositionAttentionTypes>,
    #[serde(default, deserialize_with = "deserialize_norm_type")]
    pub norm_rel_ebd: Option<NormRelEmbedTypes>,
    pub share_att_key: Option<bool>,
    pub conv_kernel_size: Option<i64>,
    pub conv_groups: Option<i64>,
    pub conv_act: Option<Activation>,
    pub pooler_dropout: Option<f64>,
    pub pooler_hidden_act: Option<Activation>,
    pub pooler_hidden_size: Option<i64>,
    pub layer_norm_eps: Option<f64>,
    pub pad_token_id: Option<i64>,
    pub relative_attention: Option<bool>,
    pub max_relative_positions: Option<i64>,
    pub embedding_size: Option<i64>,
    pub talking_head: Option<bool>,
    pub output_hidden_states: Option<bool>,
    pub output_attentions: Option<bool>,
    pub classifier_activation: Option<bool>,
    pub classifier_dropout: Option<f64>,
    pub is_decoder: Option<bool>,
    #[serde(default)]
    pub id2label: Option<HashMap<i64, String>>,
    #[serde(default)]
    pub label2id: Option<HashMap<String, i64>>,
}

// ---------------------------------------------------------------------------
// Internal DebertaConfig (used by layers, converted from V2)
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct DebertaConfig {
    pub hidden_act: Activation,
    pub attention_probs_dropout_prob: f64,
    pub hidden_dropout_prob: f64,
    pub hidden_size: i64,
    pub initializer_range: f64,
    pub intermediate_size: i64,
    pub max_position_embeddings: i64,
    pub num_attention_heads: i64,
    pub num_hidden_layers: i64,
    pub type_vocab_size: i64,
    pub vocab_size: i64,
    pub position_biased_input: Option<bool>,
    #[serde(default, deserialize_with = "deserialize_attention_type")]
    pub pos_att_type: Option<PositionAttentionTypes>,
    pub pooler_dropout: Option<f64>,
    pub pooler_hidden_act: Option<Activation>,
    pub pooler_hidden_size: Option<i64>,
    pub layer_norm_eps: Option<f64>,
    pub pad_token_id: Option<i64>,
    pub relative_attention: Option<bool>,
    pub max_relative_positions: Option<i64>,
    pub embedding_size: Option<i64>,
    pub talking_head: Option<bool>,
    pub output_hidden_states: Option<bool>,
    pub output_attentions: Option<bool>,
    pub classifier_dropout: Option<f64>,
    pub is_decoder: Option<bool>,
    pub id2label: Option<HashMap<i64, String>>,
    pub label2id: Option<HashMap<String, i64>>,
    pub share_att_key: Option<bool>,
    pub position_buckets: Option<i64>,
}

impl From<&DebertaV2Config> for DebertaConfig {
    fn from(v2: &DebertaV2Config) -> Self {
        DebertaConfig {
            hidden_act: v2.hidden_act,
            attention_probs_dropout_prob: v2.attention_probs_dropout_prob,
            hidden_dropout_prob: v2.hidden_dropout_prob,
            hidden_size: v2.hidden_size,
            initializer_range: v2.initializer_range,
            intermediate_size: v2.intermediate_size,
            max_position_embeddings: v2.max_position_embeddings,
            num_attention_heads: v2.num_attention_heads,
            num_hidden_layers: v2.num_hidden_layers,
            type_vocab_size: v2.type_vocab_size,
            vocab_size: v2.vocab_size,
            position_biased_input: v2.position_biased_input,
            pos_att_type: v2.pos_att_type.clone(),
            pooler_dropout: v2.pooler_dropout,
            pooler_hidden_act: v2.pooler_hidden_act,
            pooler_hidden_size: v2.pooler_hidden_size,
            layer_norm_eps: v2.layer_norm_eps,
            pad_token_id: v2.pad_token_id,
            relative_attention: v2.relative_attention,
            max_relative_positions: v2.max_relative_positions,
            embedding_size: v2.embedding_size,
            talking_head: v2.talking_head,
            output_hidden_states: v2.output_hidden_states,
            output_attentions: v2.output_attentions,
            classifier_dropout: v2.classifier_dropout,
            is_decoder: v2.is_decoder,
            id2label: v2.id2label.clone(),
            label2id: v2.label2id.clone(),
            share_att_key: v2.share_att_key,
            position_buckets: v2.position_buckets,
        }
    }
}
