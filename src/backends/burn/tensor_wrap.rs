use anyhow::{Result, bail};
use burn::prelude::*;

/// Dynamic-rank tensor wrapper bridging Burn's static `Tensor<B,D>` to [`crate::Gliner2Engine::Tensor`].
#[derive(Clone, Debug)]
pub enum BurnTensor<B: Backend> {
    F1(Tensor<B, 1>),
    F2(Tensor<B, 2>),
    F3(Tensor<B, 3>),
    F4(Tensor<B, 4>),
    /// Integer 2-D (input_ids, attention_mask).
    I2(Tensor<B, 2, Int>),
}

// --- unwrap helpers ---

impl<B: Backend> BurnTensor<B> {
    pub fn into_f1(self) -> Result<Tensor<B, 1>> {
        match self {
            Self::F1(t) => Ok(t),
            other => bail!("expected F1, got {}", other.tag()),
        }
    }
    pub fn as_f1(&self) -> Result<&Tensor<B, 1>> {
        match self {
            Self::F1(t) => Ok(t),
            other => bail!("expected F1, got {}", other.tag()),
        }
    }
    pub fn into_f2(self) -> Result<Tensor<B, 2>> {
        match self {
            Self::F2(t) => Ok(t),
            other => bail!("expected F2, got {}", other.tag()),
        }
    }
    pub fn as_f2(&self) -> Result<&Tensor<B, 2>> {
        match self {
            Self::F2(t) => Ok(t),
            other => bail!("expected F2, got {}", other.tag()),
        }
    }
    pub fn into_f3(self) -> Result<Tensor<B, 3>> {
        match self {
            Self::F3(t) => Ok(t),
            other => bail!("expected F3, got {}", other.tag()),
        }
    }
    pub fn as_f3(&self) -> Result<&Tensor<B, 3>> {
        match self {
            Self::F3(t) => Ok(t),
            other => bail!("expected F3, got {}", other.tag()),
        }
    }
    pub fn into_f4(self) -> Result<Tensor<B, 4>> {
        match self {
            Self::F4(t) => Ok(t),
            other => bail!("expected F4, got {}", other.tag()),
        }
    }
    pub fn as_f4(&self) -> Result<&Tensor<B, 4>> {
        match self {
            Self::F4(t) => Ok(t),
            other => bail!("expected F4, got {}", other.tag()),
        }
    }
    pub fn into_i2(self) -> Result<Tensor<B, 2, Int>> {
        match self {
            Self::I2(t) => Ok(t),
            other => bail!("expected I2, got {}", other.tag()),
        }
    }
    pub fn as_i2(&self) -> Result<&Tensor<B, 2, Int>> {
        match self {
            Self::I2(t) => Ok(t),
            other => bail!("expected I2, got {}", other.tag()),
        }
    }

    fn tag(&self) -> &'static str {
        match self {
            Self::F1(_) => "F1",
            Self::F2(_) => "F2",
            Self::F3(_) => "F3",
            Self::F4(_) => "F4",
            Self::I2(_) => "I2",
        }
    }

    /// First dimension size.
    pub fn dim0(&self) -> usize {
        match self {
            Self::F1(t) => t.dims()[0],
            Self::F2(t) => t.dims()[0],
            Self::F3(t) => t.dims()[0],
            Self::F4(t) => t.dims()[0],
            Self::I2(t) => t.dims()[0],
        }
    }

    /// Narrow along dimension 0 (float tensors only).
    pub fn narrow0(&self, start: usize, len: usize) -> Result<Self> {
        Ok(match self {
            Self::F1(t) => Self::F1(t.clone().narrow(0, start, len)),
            Self::F2(t) => Self::F2(t.clone().narrow(0, start, len)),
            Self::F3(t) => Self::F3(t.clone().narrow(0, start, len)),
            Self::F4(t) => Self::F4(t.clone().narrow(0, start, len)),
            Self::I2(_) => bail!("narrow0 not supported on I2"),
        })
    }

    /// Select a single row along dimension 0, reducing rank by 1.
    pub fn index0(&self, i: usize) -> Result<Self> {
        Ok(match self {
            Self::F2(t) => Self::F1(t.clone().narrow(0, i, 1).squeeze_dim::<1>(0)),
            Self::F3(t) => Self::F2(t.clone().narrow(0, i, 1).squeeze_dim::<2>(0)),
            Self::F4(t) => Self::F3(t.clone().narrow(0, i, 1).squeeze_dim::<3>(0)),
            other => bail!("index0 not supported on {}", other.tag()),
        })
    }
}
