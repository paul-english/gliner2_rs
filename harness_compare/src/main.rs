use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::debertav2::Config as DebertaConfig;
use gliner2::config::download_model;
use gliner2::decode::{self, Entity};
use gliner2::{Extractor, ExtractorConfig, SchemaTransformer};
use serde::Serialize;
use std::fs;
use std::time::Instant;

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

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let fixtures_path = args
        .get(1)
        .context("usage: harness_compare <fixtures.json> [model_id]")?;

    let model_id = args.get(2).map(String::as_str).unwrap_or(DEFAULT_MODEL_ID);

    let fixtures_json =
        fs::read_to_string(fixtures_path).with_context(|| format!("read {}", fixtures_path))?;
    let fixtures: Vec<Fixture> =
        serde_json::from_str(&fixtures_json).context("parse fixtures JSON")?;

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
        let labels: Vec<&str> = f.entity_types.iter().map(String::as_str).collect();

        let infer_start = Instant::now();
        let formatted = processor.format_input_for_ner(&f.text, &labels)?;
        let input_ids = Tensor::new(formatted.input_ids.clone(), &device)?.unsqueeze(0)?;
        let attention_mask = Tensor::ones(input_ids.dims(), candle_core::DType::I64, &device)?;
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
        model_id: model_id.to_string(),
        device_note: "cpu",
        load_model_ms,
        cases: cases_out,
    };

    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}
