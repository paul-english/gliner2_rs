//! Candle ↔ tch tensor conversion for the hybrid encoder path.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use std::convert::TryFrom;
use tch::{Device as TchDevice, Kind, Tensor as TchTensor};

pub fn candle_2d_int_to_tch(t: &Tensor, device: TchDevice) -> Result<TchTensor> {
    let dims = t.dims();
    anyhow::ensure!(dims.len() == 2, "expected 2D tensor");
    let (r, c) = (dims[0] as i64, dims[1] as i64);
    let flat = match t.dtype() {
        DType::U32 => {
            let v: Vec<u32> = t.flatten_all().context("flatten input_ids")?.to_vec1()?;
            v.into_iter().map(|x| x as i64).collect::<Vec<_>>()
        }
        DType::I64 => t.flatten_all().context("flatten mask")?.to_vec1()?,
        d => anyhow::bail!("unsupported int dtype for tch bridge: {d:?}"),
    };
    Ok(TchTensor::from_slice(&flat)
        .to_device(device)
        .view([r, c])
        .to_kind(Kind::Int64))
}

pub fn tch_hidden_to_candle(t: &TchTensor) -> Result<Tensor> {
    let t = t.to_device(TchDevice::Cpu).contiguous();
    let dims: Vec<usize> = t.size().iter().map(|&x| x as usize).collect();
    let flat = t.flatten(0, -1).to_kind(Kind::Float);
    let v64 = Vec::<f64>::try_from(flat).map_err(|e: tch::TchError| anyhow::anyhow!("{e}"))?;
    let v32: Vec<f32> = v64.into_iter().map(|x| x as f32).collect();
    Tensor::from_vec(v32, dims, &Device::Cpu).map_err(|e| anyhow::anyhow!(e.to_string()))
}
