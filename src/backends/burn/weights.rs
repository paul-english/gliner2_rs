use anyhow::{Context, Result, bail};
use burn::prelude::*;
use burn::tensor::TensorData;
use std::collections::HashMap;
use std::path::Path;

/// Intermediate weight storage: tensor name → raw `TensorData`.
pub struct WeightMap {
    pub data: HashMap<String, TensorData>,
}

impl WeightMap {
    /// Load all tensors from a SafeTensors file into `TensorData` (f32).
    pub fn from_safetensors(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        let st = safetensors::SafeTensors::deserialize(&bytes)
            .with_context(|| format!("deserialize {}", path.display()))?;

        let mut data = HashMap::new();
        for (name, view) in st.tensors() {
            let shape: Vec<usize> = view.shape().to_vec();
            let floats = tensor_view_to_f32(&view)?;
            data.insert(name, TensorData::new(floats, shape));
        }
        Ok(Self { data })
    }

    pub fn get(&self, key: &str) -> Result<&TensorData> {
        self.data
            .get(key)
            .with_context(|| format!("weight key not found: {key}"))
    }

    pub fn tensor1<B: Backend>(&self, key: &str, device: &B::Device) -> Result<Tensor<B, 1>> {
        Ok(Tensor::from_data(self.get(key)?.clone(), device))
    }

    pub fn tensor2<B: Backend>(&self, key: &str, device: &B::Device) -> Result<Tensor<B, 2>> {
        Ok(Tensor::from_data(self.get(key)?.clone(), device))
    }

    pub fn tensor3<B: Backend>(&self, key: &str, device: &B::Device) -> Result<Tensor<B, 3>> {
        Ok(Tensor::from_data(self.get(key)?.clone(), device))
    }
}

fn tensor_view_to_f32(view: &safetensors::tensor::TensorView<'_>) -> Result<Vec<f32>> {
    use safetensors::Dtype;
    let raw = view.data();
    match view.dtype() {
        Dtype::F32 => {
            let mut out = vec![0f32; raw.len() / 4];
            for (i, chunk) in raw.chunks_exact(4).enumerate() {
                out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
            Ok(out)
        }
        Dtype::F16 => {
            let mut out = Vec::with_capacity(raw.len() / 2);
            for chunk in raw.chunks_exact(2) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(half::f16::from_bits(bits).to_f32());
            }
            Ok(out)
        }
        Dtype::BF16 => {
            let mut out = Vec::with_capacity(raw.len() / 2);
            for chunk in raw.chunks_exact(2) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(half::bf16::from_bits(bits).to_f32());
            }
            Ok(out)
        }
        other => bail!("unsupported safetensors dtype: {other:?}"),
    }
}
