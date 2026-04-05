//! Entity throughput: legacy NER `forward` loop vs optional batched `batch_extract_entities`.
use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::debertav2::Config as DebertaConfig;
#[cfg(feature = "tch-backend")]
use gliner2::TchExtractor;
use gliner2::config::download_model;
use gliner2::decode::{self, Entity};
use gliner2::{ExtractOptions, Extractor, ExtractorConfig, SchemaTransformer};
use serde::Serialize;
use std::fs;
use std::time::Instant;
#[cfg(feature = "tch-backend")]
use tch::Device as TchDevice;

const DEFAULT_MODEL_ID: &str = "fastino/gliner2-base-v1";
/// Same labels on every sample so Python can use batch_size = N.
const THROUGHPUT_LABELS: &[&str] = &["company", "person", "product", "location", "date"];

#[derive(serde::Deserialize)]
struct Fixture {
    text: String,
}

#[derive(Serialize)]
struct ThroughputOutput {
    runner: &'static str,
    model_id: String,
    /// `candle` (full Candle stack) or `tch` (LibTorch encoder + Candle heads).
    backend: String,
    device_note: &'static str,
    mode: &'static str,
    /// Encoder batch size for `batch_extract_entities`; 1 means sequential `forward` loop.
    batch_size: usize,
    samples: usize,
    warmup_full_passes: usize,
    load_model_ms: f64,
    total_infer_ms: f64,
    samples_per_sec: f64,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Backend {
    Candle,
    #[cfg(feature = "tch-backend")]
    Tch,
}

fn parse_args() -> Result<(String, String, usize, usize, usize, Backend)> {
    let args: Vec<String> = std::env::args().collect();
    let mut fixtures_path: Option<String> = None;
    let mut model_id = DEFAULT_MODEL_ID.to_string();
    let mut samples = 64usize;
    let mut warmup = 2usize;
    let mut rust_batch_size = 1usize;
    let mut backend = Backend::Candle;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--fixtures" => {
                i += 1;
                fixtures_path = Some(args.get(i).context("--fixtures needs a path")?.clone());
            }
            "--model-id" => {
                i += 1;
                model_id = args.get(i).context("--model-id needs a value")?.clone();
            }
            "--samples" => {
                i += 1;
                samples = args
                    .get(i)
                    .context("--samples needs a value")?
                    .parse()
                    .context("parse --samples")?;
            }
            "--warmup" => {
                i += 1;
                warmup = args
                    .get(i)
                    .context("--warmup needs a value")?
                    .parse()
                    .context("parse --warmup")?;
            }
            "--rust-batch-size" => {
                i += 1;
                rust_batch_size = args
                    .get(i)
                    .context("--rust-batch-size needs a value")?
                    .parse()
                    .context("parse --rust-batch-size")?;
            }
            "--backend" => {
                i += 1;
                let b = args
                    .get(i)
                    .context("--backend needs candle or tch")?
                    .to_ascii_lowercase();
                match b.as_str() {
                    "candle" => backend = Backend::Candle,
                    #[cfg(feature = "tch-backend")]
                    "tch" => backend = Backend::Tch,
                    #[cfg(not(feature = "tch-backend"))]
                    "tch" => anyhow::bail!(
                        "backend tch requires building harness_compare with --features tch-backend"
                    ),
                    other => anyhow::bail!("unknown --backend {other:?}; use candle or tch"),
                }
            }
            other if !other.starts_with('-') && fixtures_path.is_none() => {
                fixtures_path = Some(other.to_string());
            }
            other => {
                anyhow::bail!(
                    "unknown arg {other:?}; try: harness_throughput <fixtures.json> [--samples N] [--warmup W] [--model-id ID] [--rust-batch-size B] [--backend candle|tch]"
                )
            }
        }
        i += 1;
    }

    let fixtures_path = fixtures_path.context(
        "usage: harness_throughput <fixtures.json> [--samples 64] [--warmup 2] [--model-id ID] [--rust-batch-size B] [--backend candle|tch]",
    )?;

    Ok((
        fixtures_path,
        model_id,
        samples,
        warmup,
        rust_batch_size,
        backend,
    ))
}

fn backend_str(b: Backend) -> &'static str {
    match b {
        Backend::Candle => "candle",
        #[cfg(feature = "tch-backend")]
        Backend::Tch => "tch",
    }
}

fn device_note(b: Backend) -> &'static str {
    match b {
        Backend::Candle => "cpu",
        #[cfg(feature = "tch-backend")]
        Backend::Tch => "cpu_libtorch",
    }
}

fn time_sequential<F>(
    warmup_passes: usize,
    samples: usize,
    base: &[Fixture],
    mut run_one: F,
) -> Result<f64>
where
    F: FnMut(&str) -> Result<Vec<Entity>>,
{
    for _ in 0..warmup_passes {
        for i in 0..samples {
            let text = &base[i % base.len()].text;
            let _ = run_one(text)?;
        }
    }
    let infer_start = Instant::now();
    for i in 0..samples {
        let text = &base[i % base.len()].text;
        let _ = run_one(text)?;
    }
    Ok(infer_start.elapsed().as_secs_f64() * 1000.0)
}

fn main() -> Result<()> {
    let (fixtures_path, model_id, samples, warmup_passes, rust_batch_size, backend) = parse_args()?;

    let fixtures_json =
        fs::read_to_string(&fixtures_path).with_context(|| format!("read {}", fixtures_path))?;
    let base: Vec<Fixture> = serde_json::from_str(&fixtures_json).context("parse fixtures JSON")?;
    anyhow::ensure!(!base.is_empty(), "fixtures must be non-empty");

    let labels: Vec<&str> = THROUGHPUT_LABELS.to_vec();
    let threshold = 0.5f32;

    let load_start = Instant::now();
    let files = download_model(&model_id)?;
    let device = Device::Cpu;
    let dtype = candle_core::DType::F32;

    let config: ExtractorConfig = serde_json::from_str(&fs::read_to_string(&files.config)?)?;
    let mut encoder_config: DebertaConfig =
        serde_json::from_str(&fs::read_to_string(&files.encoder_config)?)?;
    let processor = SchemaTransformer::new(files.tokenizer.to_str().unwrap())?;
    encoder_config.vocab_size = processor.tokenizer.get_vocab_size(true);

    let labels_owned: Vec<String> = labels.iter().map(|s| (*s).to_string()).collect();
    let extract_opts = ExtractOptions {
        threshold,
        format_results: false,
        include_confidence: false,
        include_spans: false,
        max_len: None,
        batch_size: rust_batch_size.max(1),
    };

    match backend {
        Backend::Candle => {
            let vb =
                unsafe { VarBuilder::from_mmaped_safetensors(&[files.weights], dtype, &device)? };
            let extractor = Extractor::load(config, encoder_config, vb)?;
            let load_model_ms = load_start.elapsed().as_secs_f64() * 1000.0;

            let run_one = |text: &str| -> Result<Vec<Entity>> {
                let formatted = processor.format_input_for_ner(text, &labels)?;
                let input_ids = Tensor::new(formatted.input_ids.clone(), &device)?.unsqueeze(0)?;
                let attention_mask =
                    Tensor::ones(input_ids.dims(), candle_core::DType::I64, &device)?;
                let scores = extractor.forward(&input_ids, &attention_mask, &formatted)?;
                Ok(decode::find_spans(
                    &scores,
                    threshold,
                    &labels,
                    text,
                    &formatted.start_offsets,
                    &formatted.end_offsets,
                )?)
            };

            if rust_batch_size <= 1 {
                let total_infer_ms = time_sequential(warmup_passes, samples, &base, run_one)?;
                let samples_per_sec = if total_infer_ms > 0.0 {
                    samples as f64 / (total_infer_ms / 1000.0)
                } else {
                    f64::INFINITY
                };
                let out = ThroughputOutput {
                    runner: "rust_throughput",
                    model_id,
                    backend: backend_str(backend).to_string(),
                    device_note: device_note(backend),
                    mode: "sequential_forward",
                    batch_size: 1,
                    samples,
                    warmup_full_passes: warmup_passes,
                    load_model_ms,
                    total_infer_ms,
                    samples_per_sec,
                };
                println!("{}", serde_json::to_string_pretty(&out)?);
                return Ok(());
            }

            let text_vec: Vec<String> = (0..samples)
                .map(|i| base[i % base.len()].text.clone())
                .collect();

            for _ in 0..warmup_passes {
                let _ = extractor.batch_extract_entities(
                    &processor,
                    &text_vec,
                    &labels_owned,
                    &extract_opts,
                )?;
            }

            let infer_start = Instant::now();
            let _ = extractor.batch_extract_entities(
                &processor,
                &text_vec,
                &labels_owned,
                &extract_opts,
            )?;
            let total_infer_ms = infer_start.elapsed().as_secs_f64() * 1000.0;
            let samples_per_sec = if total_infer_ms > 0.0 {
                samples as f64 / (total_infer_ms / 1000.0)
            } else {
                f64::INFINITY
            };

            let out = ThroughputOutput {
                runner: "rust_throughput",
                model_id,
                backend: backend_str(backend).to_string(),
                device_note: device_note(backend),
                mode: "batched_extract_entities",
                batch_size: rust_batch_size.max(1),
                samples,
                warmup_full_passes: warmup_passes,
                load_model_ms,
                total_infer_ms,
                samples_per_sec,
            };

            println!("{}", serde_json::to_string_pretty(&out)?);
        }
        #[cfg(feature = "tch-backend")]
        Backend::Tch => {
            let tch_extractor = TchExtractor::load(
                &files,
                config,
                encoder_config,
                processor.tokenizer.get_vocab_size(true),
                TchDevice::Cpu,
            )?;
            let load_model_ms = load_start.elapsed().as_secs_f64() * 1000.0;

            let run_one = |text: &str| -> Result<Vec<Entity>> {
                let formatted = processor.format_input_for_ner(text, &labels)?;
                let input_ids = Tensor::new(formatted.input_ids.clone(), &device)?.unsqueeze(0)?;
                let attention_mask =
                    Tensor::ones(input_ids.dims(), candle_core::DType::I64, &device)?;
                let scores = tch_extractor.forward(&input_ids, &attention_mask, &formatted)?;
                Ok(decode::find_spans(
                    &scores,
                    threshold,
                    &labels,
                    text,
                    &formatted.start_offsets,
                    &formatted.end_offsets,
                )?)
            };

            if rust_batch_size <= 1 {
                let total_infer_ms = time_sequential(warmup_passes, samples, &base, run_one)?;
                let samples_per_sec = if total_infer_ms > 0.0 {
                    samples as f64 / (total_infer_ms / 1000.0)
                } else {
                    f64::INFINITY
                };
                let out = ThroughputOutput {
                    runner: "rust_throughput",
                    model_id,
                    backend: backend_str(backend).to_string(),
                    device_note: device_note(backend),
                    mode: "sequential_forward",
                    batch_size: 1,
                    samples,
                    warmup_full_passes: warmup_passes,
                    load_model_ms,
                    total_infer_ms,
                    samples_per_sec,
                };
                println!("{}", serde_json::to_string_pretty(&out)?);
                return Ok(());
            }

            let text_vec: Vec<String> = (0..samples)
                .map(|i| base[i % base.len()].text.clone())
                .collect();

            for _ in 0..warmup_passes {
                let _ = tch_extractor.batch_extract_entities(
                    &processor,
                    &text_vec,
                    &labels_owned,
                    &extract_opts,
                )?;
            }

            let infer_start = Instant::now();
            let _ = tch_extractor.batch_extract_entities(
                &processor,
                &text_vec,
                &labels_owned,
                &extract_opts,
            )?;
            let total_infer_ms = infer_start.elapsed().as_secs_f64() * 1000.0;
            let samples_per_sec = if total_infer_ms > 0.0 {
                samples as f64 / (total_infer_ms / 1000.0)
            } else {
                f64::INFINITY
            };

            let out = ThroughputOutput {
                runner: "rust_throughput",
                model_id,
                backend: backend_str(backend).to_string(),
                device_note: device_note(backend),
                mode: "batched_extract_entities",
                batch_size: rust_batch_size.max(1),
                samples,
                warmup_full_passes: warmup_passes,
                load_model_ms,
                total_infer_ms,
                samples_per_sec,
            };

            println!("{}", serde_json::to_string_pretty(&out)?);
        }
    }

    Ok(())
}
