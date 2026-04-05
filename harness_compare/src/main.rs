use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::debertav2::Config as DebertaConfig;
#[cfg(feature = "tch-backend")]
use gliner2::TchExtractor;
use gliner2::config::download_model;
use gliner2::decode::{self, Entity};
use gliner2::{Extractor, ExtractorConfig, SchemaTransformer};
use serde::Serialize;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;
#[cfg(feature = "tch-backend")]
use tch::Device as TchDevice;

const DEFAULT_MODEL_ID: &str = "fastino/gliner2-base-v1";

#[derive(serde::Deserialize)]
struct Fixture {
    id: String,
    text: String,
    entity_types: Vec<String>,
    threshold: f32,
}

#[derive(Serialize)]
struct HarnessOutput {
    runner: &'static str,
    model_id: String,
    backend: String,
    device_note: &'static str,
    load_model_ms: f64,
    cases: Vec<CaseOutput>,
}

#[derive(Serialize)]
struct CaseOutput {
    id: String,
    text: String,
    entity_types: Vec<String>,
    threshold: f32,
    infer_ms: f64,
    entities: Vec<Entity>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Backend {
    Candle,
    #[cfg(feature = "tch-backend")]
    Tch,
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

struct ParsedArgs {
    fixtures: PathBuf,
    model_id: String,
    backend: Backend,
}

fn parse_args() -> Result<ParsedArgs> {
    let args: Vec<String> = std::env::args().collect();
    let mut fixtures: Option<PathBuf> = None;
    let mut model_id: Option<String> = None;
    let mut backend = Backend::Candle;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model-id" => {
                i += 1;
                model_id = Some(args.get(i).context("--model-id needs a value")?.clone());
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
            other if !other.starts_with('-') => {
                if fixtures.is_none() {
                    fixtures = Some(PathBuf::from(other));
                } else if model_id.is_none() {
                    model_id = Some(other.to_string());
                } else {
                    anyhow::bail!("unexpected positional argument {other:?}");
                }
            }
            other => anyhow::bail!("unknown argument {other:?}"),
        }
        i += 1;
    }

    let fixtures = fixtures.context(
        "usage: harness_compare <fixtures.json> [--model-id ID] [--backend candle|tch]\n\
         legacy: harness_compare <fixtures.json> [model_id]",
    )?;

    Ok(ParsedArgs {
        fixtures,
        model_id: model_id.unwrap_or_else(|| DEFAULT_MODEL_ID.to_string()),
        backend,
    })
}

fn main() -> Result<()> {
    let ParsedArgs {
        fixtures,
        model_id,
        backend,
    } = parse_args()?;

    let fixtures_json =
        fs::read_to_string(&fixtures).with_context(|| format!("read {}", fixtures.display()))?;
    let fixtures: Vec<Fixture> =
        serde_json::from_str(&fixtures_json).context("parse fixtures JSON")?;

    let load_start = Instant::now();
    let files = download_model(&model_id)?;
    let device = Device::Cpu;
    let dtype = candle_core::DType::F32;

    let config: ExtractorConfig = serde_json::from_str(&fs::read_to_string(&files.config)?)?;
    let mut encoder_config: DebertaConfig =
        serde_json::from_str(&fs::read_to_string(&files.encoder_config)?)?;
    let processor = SchemaTransformer::new(files.tokenizer.to_str().unwrap())?;
    let vocab = processor.tokenizer.get_vocab_size(true);
    encoder_config.vocab_size = vocab;

    let mut cases_out = Vec::with_capacity(fixtures.len());

    match backend {
        Backend::Candle => {
            let vb =
                unsafe { VarBuilder::from_mmaped_safetensors(&[files.weights], dtype, &device)? };
            let extractor = Extractor::load(config, encoder_config, vb)?;
            let load_model_ms = load_start.elapsed().as_secs_f64() * 1000.0;

            for f in fixtures {
                let labels: Vec<&str> = f.entity_types.iter().map(String::as_str).collect();

                let infer_start = Instant::now();
                let formatted = processor.format_input_for_ner(&f.text, &labels)?;
                let input_ids = Tensor::new(formatted.input_ids.clone(), &device)?.unsqueeze(0)?;
                let attention_mask =
                    Tensor::ones(input_ids.dims(), candle_core::DType::I64, &device)?;
                let scores = extractor.forward(&input_ids, &attention_mask, &formatted)?;
                let entities = decode::find_spans(
                    &scores,
                    f.threshold,
                    &labels,
                    &f.text,
                    &formatted.start_offsets,
                    &formatted.end_offsets,
                )?;
                let infer_ms = infer_start.elapsed().as_secs_f64() * 1000.0;

                cases_out.push(CaseOutput {
                    id: f.id,
                    text: f.text,
                    entity_types: f.entity_types,
                    threshold: f.threshold,
                    infer_ms,
                    entities,
                });
            }

            let out = HarnessOutput {
                runner: "rust",
                model_id: model_id.clone(),
                backend: backend_str(backend).to_string(),
                device_note: device_note(backend),
                load_model_ms,
                cases: cases_out,
            };
            println!("{}", serde_json::to_string_pretty(&out)?);
        }
        #[cfg(feature = "tch-backend")]
        Backend::Tch => {
            let tch_extractor =
                TchExtractor::load(&files, config, encoder_config, vocab, TchDevice::Cpu)?;
            let load_model_ms = load_start.elapsed().as_secs_f64() * 1000.0;

            for f in fixtures {
                let labels: Vec<&str> = f.entity_types.iter().map(String::as_str).collect();

                let infer_start = Instant::now();
                let formatted = processor.format_input_for_ner(&f.text, &labels)?;
                let input_ids = Tensor::new(formatted.input_ids.clone(), &device)?.unsqueeze(0)?;
                let attention_mask =
                    Tensor::ones(input_ids.dims(), candle_core::DType::I64, &device)?;
                let scores = tch_extractor.forward(&input_ids, &attention_mask, &formatted)?;
                let entities = decode::find_spans(
                    &scores,
                    f.threshold,
                    &labels,
                    &f.text,
                    &formatted.start_offsets,
                    &formatted.end_offsets,
                )?;
                let infer_ms = infer_start.elapsed().as_secs_f64() * 1000.0;

                cases_out.push(CaseOutput {
                    id: f.id,
                    text: f.text,
                    entity_types: f.entity_types,
                    threshold: f.threshold,
                    infer_ms,
                    entities,
                });
            }

            let out = HarnessOutput {
                runner: "rust",
                model_id: model_id.clone(),
                backend: backend_str(backend).to_string(),
                device_note: device_note(backend),
                load_model_ms,
                cases: cases_out,
            };
            println!("{}", serde_json::to_string_pretty(&out)?);
        }
    }

    Ok(())
}
