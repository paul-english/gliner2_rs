//! Load `safetensors` weights into `tch::Tensor` maps (no Candle).

use anyhow::{Context, Result, bail};
use safetensors::{Dtype, SafeTensors};
use std::collections::HashMap;
use std::path::Path;
use tch::{Device, Tensor};

pub struct TensorMap {
    pub tensors: HashMap<String, Tensor>,
}

pub fn load_safetensors(path: &Path, device: Device) -> Result<TensorMap> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let st = SafeTensors::deserialize(&bytes).context("parse safetensors")?;
    let mut tensors = HashMap::new();
    for name in st.names() {
        let v = st.tensor(name).with_context(|| format!("tensor {name}"))?;
        let t = view_to_tensor(&v, device).with_context(|| format!("convert {name}"))?;
        tensors.insert(name.to_string(), t);
    }
    Ok(TensorMap { tensors })
}

fn view_to_tensor(v: &safetensors::tensor::TensorView<'_>, device: Device) -> Result<Tensor> {
    let shape: Vec<i64> = v.shape().iter().map(|&x| x as i64).collect();
    match v.dtype() {
        Dtype::F32 => {
            let raw = v.data();
            if raw.len() % 4 != 0 {
                bail!("bad f32 byte length");
            }
            let mut vec = Vec::with_capacity(raw.len() / 4);
            for chunk in raw.chunks_exact(4) {
                vec.push(f32::from_le_bytes(chunk.try_into().unwrap()));
            }
            Ok(Tensor::from_slice(&vec)
                .to_device(device)
                .reshape(shape.as_slice()))
        }
        _ => bail!("unsupported safetensors dtype {:?} (need F32)", v.dtype()),
    }
}

pub struct LinearW {
    pub weight: Tensor,
    pub bias: Tensor,
}

impl LinearW {
    pub fn from_map(map: &HashMap<String, Tensor>, prefix: &str) -> Result<Self> {
        let wkey = format!("{prefix}.weight");
        let bkey = format!("{prefix}.bias");
        let weight = map
            .get(&wkey)
            .with_context(|| format!("missing {wkey}"))?
            .shallow_clone();
        let bias = map
            .get(&bkey)
            .with_context(|| format!("missing {bkey}"))?
            .shallow_clone();
        Ok(Self { weight, bias })
    }

    /// `x` `[..., in_features]` → `[..., out_features]`. Weight shape `[out, in]` (PyTorch layout).
    pub fn forward(&self, x: &Tensor) -> Tensor {
        x.matmul(&self.weight.transpose(0, 1)) + &self.bias
    }
}
