use anyhow::{Context, Result};
#[cfg(feature = "tch-backend")]
use gliner2::TchExtractor;
use gliner2::config::download_model;
use gliner2::extract::{ExtractOptions, extract_with_schema};
use gliner2::schema::ExtractionSchema;
use gliner2::{CandleExtractor, ExtractorConfig, SchemaTransformer, infer_metadata_from_schema};
use serde::Serialize;
use serde_json::Value;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;
#[cfg(feature = "tch-backend")]
use tch::Device as TchDevice;

const DEFAULT_MODEL_ID: &str = "fastino/gliner2-base-v1";

#[derive(serde::Deserialize)]
struct MtFixture {
    id: String,
    text: String,
    schema: ExtractionSchema,
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
    backend: String,
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
        "usage: harness_compare_mt <fixtures_multitask.json> [--model-id ID] [--backend candle|tch]\n\
         legacy: harness_compare_mt <fixtures.json> [model_id]",
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
    let fixtures: Vec<MtFixture> =
        serde_json::from_str(&fixtures_json).context("parse multitask fixtures JSON")?;

    let load_start = Instant::now();
    let files = download_model(&model_id)?;

    let config: ExtractorConfig = serde_json::from_str(&fs::read_to_string(&files.config)?)?;
    let processor = SchemaTransformer::new(files.tokenizer.to_str().unwrap())?;
    let vocab = processor.tokenizer.get_vocab_size(true);

    let mut cases_out = Vec::with_capacity(fixtures.len());

    match backend {
        Backend::Candle => {
            let extractor = CandleExtractor::load_cpu(&files, config, vocab)?;
            let load_model_ms = load_start.elapsed().as_secs_f64() * 1000.0;

            for f in fixtures {
                let meta = infer_metadata_from_schema(&f.schema);
                let opts = ExtractOptions {
                    threshold: f.threshold,
                    format_results: true,
                    include_confidence: false,
                    include_spans: false,
                    max_len: None,
                    batch_size: 8,
                };

                let infer_start = Instant::now();
                let result =
                    extract_with_schema(&extractor, &processor, &f.text, &f.schema, &meta, &opts)?;
                let infer_ms = infer_start.elapsed().as_secs_f64() * 1000.0;

                cases_out.push(MtCaseOutput {
                    id: f.id,
                    text: f.text,
                    threshold: f.threshold,
                    infer_ms,
                    result: serde_json::to_value(&result)?,
                });
            }

            let out = MtHarnessOutput {
                runner: "rust_mt",
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
            let tch_extractor = TchExtractor::load(&files, config, vocab, TchDevice::Cpu)?;
            let load_model_ms = load_start.elapsed().as_secs_f64() * 1000.0;

            for f in fixtures {
                let meta = infer_metadata_from_schema(&f.schema);
                let opts = ExtractOptions {
                    threshold: f.threshold,
                    format_results: true,
                    include_confidence: false,
                    include_spans: false,
                    max_len: None,
                    batch_size: 8,
                };

                let infer_start = Instant::now();
                let result = extract_with_schema(
                    &tch_extractor,
                    &processor,
                    &f.text,
                    &f.schema,
                    &meta,
                    &opts,
                )?;
                let infer_ms = infer_start.elapsed().as_secs_f64() * 1000.0;

                cases_out.push(MtCaseOutput {
                    id: f.id,
                    text: f.text,
                    threshold: f.threshold,
                    infer_ms,
                    result: serde_json::to_value(&result)?,
                });
            }

            let out = MtHarnessOutput {
                runner: "rust_mt",
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
