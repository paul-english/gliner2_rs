use anyhow::{Context, Result};
use candle_core::Device;
use candle_nn::VarBuilder;
use candle_transformers::models::debertav2::Config as DebertaConfig;
use gliner2::config::download_model;
use gliner2::extract::{ExtractOptions, extract_with_schema};
use gliner2::{Extractor, ExtractorConfig, SchemaTransformer, infer_metadata_from_schema};
use serde::Serialize;
use serde_json::Value;
use std::fs;
use std::time::Instant;

const DEFAULT_MODEL_ID: &str = "fastino/gliner2-base-v1";

#[derive(serde::Deserialize)]
struct MtFixture {
    id: String,
    text: String,
    schema: Value,
    #[serde(default = "default_threshold")]
    threshold: f32,
}

fn default_threshold() -> f32 {
    0.5
}

#[derive(Serialize)]
struct MtHarnessOutput {
    runner: &'static str,
    model_id: String,
    device_note: &'static str,
    load_model_ms: f64,
    cases: Vec<MtCaseOutput>,
}

#[derive(Serialize)]
struct MtCaseOutput {
    id: String,
    text: String,
    threshold: f32,
    infer_ms: f64,
    result: Value,
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let fixtures_path = args
        .get(1)
        .context("usage: harness_compare_mt <fixtures_multitask.json> [model_id]")?;

    let model_id = args.get(2).map(String::as_str).unwrap_or(DEFAULT_MODEL_ID);

    let fixtures_json =
        fs::read_to_string(fixtures_path).with_context(|| format!("read {}", fixtures_path))?;
    let fixtures: Vec<MtFixture> =
        serde_json::from_str(&fixtures_json).context("parse multitask fixtures JSON")?;

    let load_start = Instant::now();
    let files = download_model(model_id)?;
    let device = Device::Cpu;
    let dtype = candle_core::DType::F32;

    let config: ExtractorConfig = serde_json::from_str(&fs::read_to_string(&files.config)?)?;
    let mut encoder_config: DebertaConfig =
        serde_json::from_str(&fs::read_to_string(&files.encoder_config)?)?;
    let processor = SchemaTransformer::new(files.tokenizer.to_str().unwrap())?;
    encoder_config.vocab_size = processor.tokenizer.get_vocab_size(true);

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[files.weights], dtype, &device)? };
    let extractor = Extractor::load(config, encoder_config, vb)?;
    let load_model_ms = load_start.elapsed().as_secs_f64() * 1000.0;

    let mut cases_out = Vec::with_capacity(fixtures.len());

    for f in fixtures {
        let meta = infer_metadata_from_schema(&f.schema);
        let opts = ExtractOptions {
            threshold: f.threshold,
            format_results: true,
            include_confidence: false,
            include_spans: false,
            max_len: None,
        };

        let infer_start = Instant::now();
        let result = extract_with_schema(&extractor, &processor, &f.text, &f.schema, &meta, &opts)?;
        let infer_ms = infer_start.elapsed().as_secs_f64() * 1000.0;

        cases_out.push(MtCaseOutput {
            id: f.id,
            text: f.text,
            threshold: f.threshold,
            infer_ms,
            result,
        });
    }

    let out = MtHarnessOutput {
        runner: "rust_mt",
        model_id: model_id.to_string(),
        device_note: "cpu",
        load_model_ms,
        cases: cases_out,
    };

    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}
