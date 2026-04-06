// Copyright 2019-2022 Guillaume Becquin
// Copyright 2020 Microsoft and the HuggingFace Inc. team.
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

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
use tch::nn::{Embedding, ModuleT};
use tch::{Kind, Scalar, Tensor};

// ---------------------------------------------------------------------------
// Activation functions
// ---------------------------------------------------------------------------

fn _gelu(x: &Tensor) -> Tensor {
    x * 0.5 * (1.0 + (x / ((2.0_f64).sqrt())).erf())
}

fn _relu(x: &Tensor) -> Tensor {
    x.relu()
}

fn _swish(x: &Tensor) -> Tensor {
    x * x.sigmoid()
}

fn _mish(x: &Tensor) -> Tensor {
    x * (x.softplus().tanh())
}

fn _gelu_new(x: &Tensor) -> Tensor {
    x * 0.5 * (((x.pow_tensor_scalar(3.0f64) * 0.044715 + x) * ((2f64 / PI).sqrt())).tanh() + 1)
}

fn _tanh(x: &Tensor) -> Tensor {
    x.tanh()
}

fn _identity(x: &Tensor) -> Tensor {
    x.shallow_clone()
}

pub struct TensorFunction(Box<fn(&Tensor) -> Tensor>);

impl TensorFunction {
    pub fn new(fun: Box<fn(&Tensor) -> Tensor>) -> Self {
        Self(fun)
    }

    pub fn get_fn(&self) -> &fn(&Tensor) -> Tensor {
        &self.0
    }
}

impl std::fmt::Debug for TensorFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        write!(f, "TensorFunction")
    }
}

#[allow(non_camel_case_types)]
#[derive(Clone, Debug, Serialize, Deserialize, Copy)]
pub enum Activation {
    gelu,
    relu,
    swish,
    mish,
    gelu_new,
    tanh,
    identity,
}

impl Activation {
    pub fn get_function(&self) -> TensorFunction {
        TensorFunction::new(Box::new(match self {
            Activation::gelu => _gelu,
            Activation::relu => _relu,
            Activation::swish => _swish,
            Activation::gelu_new => _gelu_new,
            Activation::mish => _mish,
            Activation::tanh => _tanh,
            Activation::identity => _identity,
        }))
    }
}

// ---------------------------------------------------------------------------
// Dropout
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct Dropout {
    dropout_prob: f64,
}

impl Dropout {
    pub fn new(p: f64) -> Dropout {
        Dropout { dropout_prob: p }
    }
}

impl ModuleT for Dropout {
    fn forward_t(&self, input: &Tensor, train: bool) -> Tensor {
        input.dropout(self.dropout_prob, train)
    }
}

#[derive(Debug)]
pub struct XDropout {
    dropout_prob: f64,
}

impl XDropout {
    pub fn new(p: f64) -> XDropout {
        XDropout { dropout_prob: p }
    }
}

impl ModuleT for XDropout {
    fn forward_t(&self, input: &Tensor, train: bool) -> Tensor {
        if train {
            let mask = (Tensor::ones([1], (input.kind(), input.device()))
                - input
                    .empty_like()
                    .bernoulli_float_(1_f64 - self.dropout_prob))
            .to_kind(Kind::Bool);

            input.masked_fill(&mask, 0) / (1_f64 - self.dropout_prob)
        } else {
            input.shallow_clone()
        }
    }
}

// ---------------------------------------------------------------------------
// Kind utilities
// ---------------------------------------------------------------------------

pub(crate) fn get_min(kind: Kind) -> Result<Scalar> {
    Ok(match kind {
        Kind::Uint8 => Scalar::int(u8::MIN.into()),
        Kind::Int8 => Scalar::int(i8::MIN.into()),
        Kind::Int16 => Scalar::int(i16::MIN.into()),
        Kind::Int => Scalar::int(i32::MIN.into()),
        Kind::Int64 => Scalar::int(i64::MIN),
        Kind::Half => Scalar::float(half::f16::MIN.into()),
        Kind::Float => Scalar::float(f32::MIN.into()),
        Kind::BFloat16 => Scalar::float(half::bf16::MIN.into()),
        Kind::Double => Scalar::float(f64::MIN),
        _ => bail!("Type not supported: attempted to get min for {kind:?}"),
    })
}

pub fn x_softmax(input: &Tensor, mask: &Tensor, dim: i64) -> Tensor {
    let inverse_mask = ((1 - mask) as Tensor).to_kind(Kind::Bool);
    input
        .masked_fill(&inverse_mask, get_min(input.kind()).unwrap())
        .softmax(dim, input.kind())
        .masked_fill(&inverse_mask, 0.0)
}

// ---------------------------------------------------------------------------
// Embedding utilities
// ---------------------------------------------------------------------------

pub fn process_ids_embeddings_pair(
    input_ids: Option<&Tensor>,
    input_embeddings: Option<&Tensor>,
    embeddings_matrix: &Embedding,
) -> Result<(Option<Tensor>, Vec<i64>, tch::Device)> {
    Ok(match (input_ids, input_embeddings) {
        (Some(_), Some(_)) => {
            bail!("Only one of input ids or input embeddings may be set");
        }
        (Some(input_value), None) => (
            Some(input_value.apply(embeddings_matrix)),
            input_value.size(),
            input_value.device(),
        ),
        (None, Some(embeds)) => {
            let size = vec![embeds.size()[0], embeds.size()[1]];
            (None, size, embeds.device())
        }
        (None, None) => {
            bail!("At least one of input ids or input embeddings must be set");
        }
    })
}

pub fn get_shape_and_device_from_ids_embeddings_pair(
    input_ids: Option<&Tensor>,
    input_embeddings: Option<&Tensor>,
) -> Result<(Vec<i64>, tch::Device)> {
    Ok(match (input_ids, input_embeddings) {
        (Some(_), Some(_)) => {
            bail!("Only one of input ids or input embeddings may be set");
        }
        (Some(input_value), None) => (input_value.size(), input_value.device()),
        (None, Some(embeds)) => {
            let size = vec![embeds.size()[0], embeds.size()[1]];
            (size, embeds.device())
        }
        (None, None) => {
            bail!("At least one of input ids or input embeddings must be set");
        }
    })
}
