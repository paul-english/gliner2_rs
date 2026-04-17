// DeBERTa V2 configuration (serde-compatible with HuggingFace config.json).
// Duplicated from tch backend for independence.

use serde::{Deserialize, Serialize, de};
use std::fmt;
use std::str::FromStr;

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
            "p2c" => Ok(Self::p2c),
            "c2p" => Ok(Self::c2p),
            "p2p" => Ok(Self::p2p),
            _ => Err(format!("unknown pos_att_type: {s}")),
        }
    }
}

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
        Ok(Self { types })
    }
}

impl PositionAttentionTypes {
    pub fn has(&self, t: PositionAttentionType) -> bool {
        self.types.iter().any(|x| *x == t)
    }
    pub fn len(&self) -> usize {
        self.types.len()
    }
}

pub fn deserialize_attention_type<'de, D>(
    deserializer: D,
) -> Result<Option<PositionAttentionTypes>, D::Error>
where
    D: de::Deserializer<'de>,
{
    struct V;
    impl<'de> de::Visitor<'de> for V {
        type Value = PositionAttentionTypes;
        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("string or sequence")
        }
        fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> {
            Ok(PositionAttentionTypes::from_str(v).unwrap())
        }
        fn visit_seq<A: de::SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
            let mut types = vec![];
            while let Some(s) = seq.next_element::<String>()? {
                types.push(PositionAttentionType::from_str(&s).unwrap());
            }
            Ok(PositionAttentionTypes { types })
        }
    }
    deserializer.deserialize_any(V).map(Some)
}

#[allow(non_camel_case_types)]
#[derive(Clone, Debug, Serialize, Deserialize, Copy, PartialEq, Eq)]
pub enum NormRelEmbedType {
    layer_norm,
}

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
            .filter(|p| !p.is_empty())
            .map(|p| match p {
                "layer_norm" => Ok(NormRelEmbedType::layer_norm),
                other => Err(format!("unknown norm_rel_ebd: {other}")),
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { types })
    }
}

impl NormRelEmbedTypes {
    pub fn has(&self, t: NormRelEmbedType) -> bool {
        self.types.iter().any(|x| *x == t)
    }
}

pub fn deserialize_norm_type<'de, D>(
    deserializer: D,
) -> Result<Option<NormRelEmbedTypes>, D::Error>
where
    D: de::Deserializer<'de>,
{
    struct V;
    impl<'de> de::Visitor<'de> for V {
        type Value = NormRelEmbedTypes;
        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("string or sequence")
        }
        fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> {
            Ok(NormRelEmbedTypes::from_str(v).unwrap())
        }
        fn visit_seq<A: de::SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
            let mut types = vec![];
            while let Some(s) = seq.next_element::<String>()? {
                types.push(match s.as_str() {
                    "layer_norm" => NormRelEmbedType::layer_norm,
                    _ => return Err(de::Error::custom(format!("unknown norm type: {s}"))),
                });
            }
            Ok(NormRelEmbedTypes { types })
        }
    }
    deserializer.deserialize_any(V).map(Some)
}

#[allow(non_camel_case_types)]
#[derive(Clone, Debug, Serialize, Deserialize, Copy)]
pub enum Activation {
    gelu,
    relu,
    swish,
    tanh,
    gelu_new,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DebertaV2Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub hidden_act: Activation,
    pub attention_probs_dropout_prob: f64,
    pub hidden_dropout_prob: f64,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub position_buckets: Option<i64>,
    pub num_attention_heads: usize,
    pub type_vocab_size: usize,
    #[serde(default)]
    pub position_biased_input: Option<bool>,
    #[serde(default, deserialize_with = "deserialize_attention_type")]
    pub pos_att_type: Option<PositionAttentionTypes>,
    #[serde(default, deserialize_with = "deserialize_norm_type")]
    pub norm_rel_ebd: Option<NormRelEmbedTypes>,
    #[serde(default)]
    pub share_att_key: Option<bool>,
    #[serde(default)]
    pub conv_kernel_size: Option<usize>,
    #[serde(default)]
    pub conv_groups: Option<usize>,
    #[serde(default)]
    pub conv_act: Option<Activation>,
    #[serde(default)]
    pub layer_norm_eps: Option<f64>,
    #[serde(default)]
    pub pad_token_id: Option<usize>,
    #[serde(default)]
    pub relative_attention: Option<bool>,
    #[serde(default)]
    pub max_relative_positions: Option<i64>,
    #[serde(default)]
    pub embedding_size: Option<usize>,
    #[serde(default)]
    pub talking_head: Option<bool>,
    #[serde(default)]
    pub output_hidden_states: Option<bool>,
    #[serde(default)]
    pub output_attentions: Option<bool>,
}

impl DebertaV2Config {
    pub fn embedding_size(&self) -> usize {
        self.embedding_size.unwrap_or(self.hidden_size)
    }
    pub fn layer_norm_eps(&self) -> f64 {
        self.layer_norm_eps.unwrap_or(1e-7)
    }
    pub fn position_biased_input(&self) -> bool {
        self.position_biased_input.unwrap_or(true)
    }
    pub fn relative_attention(&self) -> bool {
        self.relative_attention.unwrap_or(false)
    }
}
