mod labelstudio;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
#[cfg(feature = "candle")]
use gliner2::CandleExtractor;
use gliner2::SchemaInfo;
use gliner2::config::{ModelFiles, download_model};
use gliner2::{
    BatchSchemaMode, ExtractOptions, ExtractorConfig, SchemaTransformer, batch_extract_streaming,
    infer_metadata_from_schema,
};
#[cfg(feature = "tch")]
use gliner2::{TchExtractor, parse_tch_device};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde_json::{Value, json};
use std::fs;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(name = "gliner2", version, about = "Gliner2 CLI")]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Hugging Face model id
    #[arg(long, default_value = "fastino/gliner2-base-v1", global = true)]
    model: String,

    /// Offline layout directory
    #[arg(long, global = true)]
    model_dir: Option<PathBuf>,

    /// Explicit path to config.json
    #[arg(long, global = true)]
    config: Option<PathBuf>,

    /// Explicit path to encoder_config/config.json
    #[arg(long, global = true)]
    encoder_config: Option<PathBuf>,

    /// Explicit path to tokenizer.json
    #[arg(long, global = true)]
    tokenizer: Option<PathBuf>,

    /// Explicit path to model.safetensors
    #[arg(long, global = true)]
    weights: Option<PathBuf>,

    /// Backend (candle or tch)
    #[arg(long, env = "GLINER2_BACKEND", global = true)]
    backend: Option<String>,

    /// LibTorch device when using `--backend tch` (ignored for candle): cpu, cuda, cuda:N, mps, vulkan, auto.
    #[arg(long, env = "GLINER2_DEVICE", default_value = "cpu", global = true)]
    device: String,

    /// Log level (off, error, warn, info, debug, trace)
    #[arg(long, default_value = "info", global = true)]
    log_level: String,

    // Inference flags
    #[arg(long, default_value_t = 0.5, global = true)]
    threshold: f32,

    #[arg(long, global = true)]
    max_len: Option<usize>,

    #[arg(long, global = true)]
    include_confidence: bool,

    #[arg(long, global = true)]
    include_spans: bool,

    #[arg(long, default_value_t = true, action = clap::ArgAction::Set, global = true)]
    format_results: bool,

    #[arg(long, global = true)]
    raw: bool,

    #[arg(long, default_value_t = 8, global = true)]
    batch_size: usize,

    /// Number of parallel engine instances (1 = sequential, >1 = parallel).
    /// With tch + cuda, workers are assigned to cuda:0, cuda:1, … automatically.
    /// Each worker loads a full copy of the model weights.
    #[arg(long, default_value_t = 1, global = true)]
    num_workers: usize,

    /// Field containing document text in JSON / JSONL records
    #[arg(long, default_value = "text", global = true)]
    text_field: String,

    /// Field to pass through as record id when present
    #[arg(long, default_value = "id", global = true)]
    id_field: String,

    /// Plain text: full (whole file) or line (one record per non-empty line)
    #[arg(long, default_value = "full", global = true)]
    text_split: String,

    /// Output path (default: stdout)
    #[arg(short, long, global = true)]
    output: Option<PathBuf>,

    /// Pretty-print JSON (only if output can be buffered)
    #[arg(long, global = true)]
    pretty: bool,

    /// Send results to a Label Studio project as pre-annotations.
    /// Requires LABEL_STUDIO_URL and LABEL_STUDIO_API_KEY env vars.
    #[arg(long, global = true)]
    labelstudio: Option<String>,
}

#[derive(Subcommand)]
enum Commands {
    /// Configure backend and download libtorch
    Setup {
        /// Skip interactive prompts
        #[arg(long)]
        non_interactive: bool,
        /// Backend: candle or tch
        #[arg(long)]
        backend: Option<String>,
        /// LibTorch variant: cpu, cu128, cu130, rocm62
        #[arg(long)]
        variant: Option<String>,
    },
    /// Show current configuration and backend status
    Status,
    /// Named-entity extraction
    Entities {
        /// Repeatable entity type name
        #[arg(long)]
        label: Vec<String>,
        /// JSON array of names or object form (name -> description)
        #[arg(long)]
        labels_json: Option<PathBuf>,
        /// Input file or "-" for stdin
        input: String,
    },
    /// Text classification
    Classify {
        /// Required classification task name
        #[arg(long)]
        task: String,
        /// Repeatable class label
        #[arg(long)]
        label: Vec<String>,
        /// Array of labels or object (label -> description)
        #[arg(long)]
        labels_json: Option<PathBuf>,
        /// Multi-label classification
        #[arg(long)]
        multi_label: bool,
        /// Per-task classifier threshold
        #[arg(long, default_value_t = 0.5)]
        cls_threshold: f32,
        /// Input file or "-" for stdin
        input: String,
    },
    /// Relation extraction
    Relations {
        /// Repeatable relation type name
        #[arg(long)]
        relation: Vec<String>,
        /// JSON array of names or object form
        #[arg(long)]
        relations_json: Option<PathBuf>,
        /// Input file or "-" for stdin
        input: String,
    },
    /// Structured JSON / field extraction
    Json {
        /// JSON file: object mapping structure name -> array of field specs
        #[arg(long)]
        structures: Option<PathBuf>,
        /// Same object inline
        #[arg(long)]
        structures_json: Option<String>,
        /// Input file or "-" for stdin
        input: String,
    },
    /// Multitask: full engine schema in one pass
    Run {
        /// Full engine multitask schema
        #[arg(long)]
        schema_file: PathBuf,
        /// Input file or "-" for stdin
        input: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Setup {
            non_interactive,
            ref backend,
            ref variant,
        } => gliner2::setup::run_setup(non_interactive, backend.as_deref(), variant.as_deref()),
        Commands::Status => show_status(),
        _ => {
            let backend = resolve_backend(&cli);
            init_tracing(&cli.log_level);
            run(cli, &backend)
        }
    }
}

fn resolve_backend(cli: &Cli) -> String {
    match &cli.backend {
        Some(b) => b.clone(),
        None => {
            if let Ok(Some(cfg)) = gliner2::setup::load_config() {
                cfg.backend.default.clone()
            } else {
                "candle".to_string()
            }
        }
    }
}

fn init_tracing(log_level: &str) {
    let level = match log_level.to_lowercase().as_str() {
        "off" => tracing::Level::ERROR,
        "error" => tracing::Level::ERROR,
        "warn" => tracing::Level::WARN,
        "info" => tracing::Level::INFO,
        "debug" => tracing::Level::DEBUG,
        "trace" => tracing::Level::TRACE,
        _ => tracing::Level::INFO,
    };
    tracing_subscriber::fmt().with_max_level(level).init();
}

fn show_status() -> Result<()> {
    let candle_available = cfg!(feature = "candle");
    let tch_compiled = cfg!(feature = "tch");

    eprintln!("gliner2 status\n");
    eprintln!("Compiled backends:");
    eprintln!("  candle: {}", if candle_available { "yes" } else { "no" });
    eprintln!("  tch:    {}", if tch_compiled { "yes" } else { "no" });

    match gliner2::setup::load_config()? {
        Some(cfg) => {
            eprintln!("\nConfig: {}", gliner2::setup::config_path()?.display());
            eprintln!("  default backend: {}", cfg.backend.default);
            if let Some(tch) = &cfg.tch {
                eprintln!("  tch variant:     {}", tch.variant);
                eprintln!("  libtorch path:   {}", tch.lib_path.display());
                eprintln!("  libtorch ver:    {}", tch.libtorch_version);
                let lib_exists = tch.lib_path.exists();
                eprintln!(
                    "  libtorch present: {}",
                    if lib_exists {
                        "yes"
                    } else {
                        "NO — run `gliner2 setup`"
                    }
                );
            }
        }
        None => {
            eprintln!("\nNo config found. Run `gliner2 setup` to configure.");
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Input / Output / Engine types
// ---------------------------------------------------------------------------

struct Record {
    id: Option<Value>,
    text: String,
}

enum Input {
    Stdin, // FIXME needs to be easily converted to StreamingRecord
    Json(PathBuf),
    Jsonl(PathBuf),
    Parquet(PathBuf),
    ParquetGlob(Vec<PathBuf>),
    PlainText(String),
    JsonText(String),
}

enum Output {
    Jsonl,
    Parquet(PathBuf),
}

impl Output {
    fn resolve(input: &Input, cli: &Cli) -> Self {
        let is_parquet_input = matches!(input, Input::Parquet(_) | Input::ParquetGlob(_));
        if is_parquet_input {
            if let Some(ref path) = cli.output {
                if path.extension().and_then(|e| e.to_str()) == Some("parquet") {
                    return Output::Parquet(path.clone());
                }
                if path.is_dir() || !path.to_string_lossy().contains('.') {
                    return Output::Parquet(path.clone());
                }
            }
        }
        Output::Jsonl
    }
}

impl Input {
    fn parse(s: &str) -> Result<Self> {
        if s == "-" {
            return Ok(Input::Stdin);
        }
        // Check for glob patterns
        if s.contains('*') || s.contains('?') {
            let mut paths: Vec<PathBuf> = glob::glob(s)
                .map_err(|e| anyhow::anyhow!("Invalid glob pattern: {e}"))?
                .filter_map(|r| r.ok())
                .collect();
            if paths.is_empty() {
                anyhow::bail!("No files matched glob pattern: {s}");
            }
            paths.sort();
            if paths
                .iter()
                .all(|p| p.extension().and_then(|e| e.to_str()) == Some("parquet"))
            {
                return Ok(Input::ParquetGlob(paths));
            }
            anyhow::bail!(
                "Glob pattern matched non-parquet files; glob input is only supported for .parquet files"
            );
        }
        let path = Path::new(s);
        Ok(match path.extension().and_then(|e| e.to_str()) {
            Some("json") => Input::Json(path.to_path_buf()),
            Some("jsonl") => Input::Jsonl(path.to_path_buf()),
            Some("parquet") => Input::Parquet(path.to_path_buf()),
            Some("txt") | Some("md") => {
                Input::PlainText(fs::read_to_string(path).unwrap_or_else(|_| s.to_string()))
            }
            _ => {
                if path.exists() {
                    Input::PlainText(fs::read_to_string(path).unwrap_or_else(|_| s.to_string()))
                } else if s.starts_with('{') || s.starts_with('[') {
                    Input::JsonText(s.to_string())
                } else {
                    Input::PlainText(s.to_string())
                }
            }
        })
    }

    /// Bail if a `.jsonl` file is actually a JSON array.
    fn validate_jsonl(path: &Path) -> Result<()> {
        let file = fs::File::open(path)?;
        let mut reader = io::BufReader::new(file);
        let mut buf = [0u8; 1];
        loop {
            match std::io::Read::read(&mut reader, &mut buf) {
                Ok(0) => return Ok(()),
                Ok(_) => {
                    if buf[0].is_ascii_whitespace() {
                        continue;
                    }
                    if buf[0] == b'[' {
                        anyhow::bail!(
                            "{}: file starts with '[' (JSON array), but has .jsonl extension.\n\
                             JSONL files must have one JSON object per line.\n\
                             Rename to .json to treat as a JSON array.",
                            path.display()
                        );
                    }
                    return Ok(());
                }
                Err(e) => return Err(e.into()),
            }
        }
    }

    fn iter_records(self, cli: &Cli) -> Result<Vec<Record>> {
        match self {
            Input::Parquet(path) => read_parquet_records(&path, cli),
            Input::ParquetGlob(paths) => {
                let mut records = Vec::new();
                for path in &paths {
                    records.extend(read_parquet_records(path, cli)?);
                }
                Ok(records)
            }
            Input::Json(path) => {
                let val: Value =
                    serde_json::from_reader(io::BufReader::new(fs::File::open(&path)?))?;
                if let Some(arr) = val.as_array() {
                    arr.iter().map(|v| val_to_record(v, cli)).collect()
                } else {
                    Ok(vec![val_to_record(&val, cli)?])
                }
            }
            Input::Jsonl(path) => {
                Self::validate_jsonl(&path)?;
                read_jsonl(io::BufReader::new(fs::File::open(&path)?), cli)
            }
            Input::Stdin => read_jsonl(io::BufReader::new(io::stdin()), cli),
            Input::JsonText(text) => {
                let val: Value = serde_json::from_str(&text)?;
                if let Some(arr) = val.as_array() {
                    arr.iter().map(|v| val_to_record(v, cli)).collect()
                } else {
                    Ok(vec![val_to_record(&val, cli)?])
                }
            }
            Input::PlainText(text) => {
                if cli.text_split == "line" {
                    Ok(text
                        .lines()
                        .filter(|l| !l.trim().is_empty())
                        .map(|l| Record {
                            id: None,
                            text: l.to_string(),
                        })
                        .collect())
                } else {
                    Ok(vec![Record { id: None, text }])
                }
            }
        }
    }
}

#[allow(clippy::large_enum_variant)]
enum Engine {
    #[cfg(feature = "candle")]
    Candle(CandleExtractor),
    #[cfg(feature = "tch")]
    Tch(TchExtractor),
}

// ---------------------------------------------------------------------------
// Inference
// ---------------------------------------------------------------------------

fn resolve_model_files(cli: &Cli) -> Result<ModelFiles> {
    if let Some(dir) = &cli.model_dir {
        return Ok(ModelFiles {
            config: dir.join("config.json"),
            encoder_config: dir.join("encoder_config/config.json"),
            tokenizer: dir.join("tokenizer.json"),
            weights: dir.join("model.safetensors"),
        });
    }

    if let (Some(c), Some(e), Some(t), Some(w)) = (
        &cli.config,
        &cli.encoder_config,
        &cli.tokenizer,
        &cli.weights,
    ) {
        return Ok(ModelFiles {
            config: c.clone(),
            encoder_config: e.clone(),
            tokenizer: t.clone(),
            weights: w.clone(),
        });
    }

    download_model(&cli.model)
}

fn run(cli: Cli, backend: &str) -> Result<()> {
    // Validate Label Studio config early (before model load)
    let ls_config = if let Some(ref project_name) = cli.labelstudio {
        let url = std::env::var("LABEL_STUDIO_URL")
            .context("--labelstudio requires LABEL_STUDIO_URL env var")?;
        let api_key = std::env::var("LABEL_STUDIO_API_KEY")
            .context("--labelstudio requires LABEL_STUDIO_API_KEY env var")?;
        Some((project_name.clone(), url, api_key))
    } else {
        None
    };

    let files = resolve_model_files(&cli)?;

    let config: ExtractorConfig = serde_json::from_str(&fs::read_to_string(&files.config)?)?;

    let processor = SchemaTransformer::new(files.tokenizer.to_str().unwrap())?;
    let vocab = processor.tokenizer.get_vocab_size(true);

    let num_workers = cli.num_workers.max(1);
    if num_workers > 1 {
        tracing::info!(
            num_workers,
            batch_size = cli.batch_size,
            device = %cli.device,
            backend,
            "multi-worker parallel extraction enabled"
        );
    }
    let engines = create_engines(&cli, backend, &files, &config, vocab, num_workers)?;

    let input_path = match &cli.command {
        Commands::Entities { input, .. } => input,
        Commands::Classify { input, .. } => input,
        Commands::Relations { input, .. } => input,
        Commands::Json { input, .. } => input,
        Commands::Run { input, .. } => input,
        Commands::Setup { .. } | Commands::Status => unreachable!(),
    };

    let input = Input::parse(input_path)?;
    let output_format = Output::resolve(&input, &cli);

    let (schema, meta) = build_schema_and_meta(&cli.command)?;

    // When Label Studio output is active, force include_spans and include_confidence
    // so that char offsets are available for NER/relation pre-annotations.
    let opts = ExtractOptions {
        threshold: cli.threshold,
        format_results: !cli.raw && cli.format_results,
        include_confidence: cli.include_confidence || ls_config.is_some(),
        include_spans: cli.include_spans || ls_config.is_some(),
        max_len: cli.max_len,
        batch_size: cli.batch_size,
    };

    // Set up Label Studio project if requested
    let ls_state = if let Some((ref project_name, ref url, ref api_key)) = ls_config {
        let info = SchemaInfo::from_value(&schema);
        let label_config = labelstudio::generate_label_config(&info);
        let project_id =
            labelstudio::create_or_get_project(url, api_key, project_name, &label_config)?;
        Some((url.clone(), api_key.clone(), project_id, info))
    } else {
        None
    };

    // Multi-file parquet: process each file independently, streaming results
    if let Input::ParquetGlob(ref paths) = input {
        if let Output::Parquet(ref out_dir) = output_format {
            fs::create_dir_all(out_dir)?;
            if num_workers > 1 {
                // Process parquet files in parallel — one engine per file (round-robin).
                // Each file uses a single engine to avoid nested rayon parallelism
                // that would cause thread contention.
                use rayon::prelude::*;
                paths
                    .par_iter()
                    .enumerate()
                    .map(|(file_idx, in_path)| -> Result<()> {
                        let records = read_parquet_records(in_path, &cli)?;
                        if records.is_empty() {
                            return Ok(());
                        }
                        let texts: Vec<String> = records.iter().map(|r| r.text.clone()).collect();
                        let out_path = out_dir
                            .join(in_path.file_name().context("input file has no filename")?);
                        let mut writer = create_parquet_writer(&out_path, &cli)?;
                        let engine = &engines[file_idx % engines.len()];
                        run_extract_streaming(
                            engine,
                            &processor,
                            &texts,
                            &schema,
                            &meta,
                            &opts,
                            |offset, batch| {
                                write_parquet_batch(
                                    &mut writer,
                                    &records[offset..offset + batch.len()],
                                    &batch,
                                    &cli,
                                )
                            },
                        )?;
                        writer.close()?;
                        Ok(())
                    })
                    .collect::<Result<Vec<_>>>()?;
            } else {
                for in_path in paths {
                    let records = read_parquet_records(in_path, &cli)?;
                    if records.is_empty() {
                        continue;
                    }
                    let texts: Vec<String> = records.iter().map(|r| r.text.clone()).collect();
                    let out_path =
                        out_dir.join(in_path.file_name().context("input file has no filename")?);
                    let mut writer = create_parquet_writer(&out_path, &cli)?;
                    run_extract_dispatch(
                        &engines,
                        &processor,
                        &texts,
                        &schema,
                        &meta,
                        &opts,
                        |offset, batch| {
                            write_parquet_batch(
                                &mut writer,
                                &records[offset..offset + batch.len()],
                                &batch,
                                &cli,
                            )
                        },
                    )?;
                    writer.close()?;
                }
            }
            return Ok(());
        }
    }

    // Single-input path (or JSONL output for any input)
    let records = input.iter_records(&cli)?;
    if records.is_empty() {
        return Ok(());
    }

    let texts: Vec<String> = records.iter().map(|r| r.text.clone()).collect();

    // Accumulate Label Studio tasks alongside normal output when --labelstudio is active
    let mut ls_tasks: Vec<Value> = Vec::new();

    match output_format {
        Output::Parquet(ref path) => {
            let mut writer = create_parquet_writer(path, &cli)?;
            run_extract_dispatch(
                &engines,
                &processor,
                &texts,
                &schema,
                &meta,
                &opts,
                |offset, batch| {
                    if let Some((_, _, _, ref info)) = ls_state {
                        for (i, result) in batch.iter().enumerate() {
                            ls_tasks.push(labelstudio::convert_result_to_task(
                                &records[offset + i].text,
                                result,
                                info,
                            ));
                        }
                    }
                    write_parquet_batch(
                        &mut writer,
                        &records[offset..offset + batch.len()],
                        &batch,
                        &cli,
                    )
                },
            )?;
            writer.close()?;
        }
        Output::Jsonl => {
            let mut out_writer = create_jsonl_writer(&cli)?;
            let use_pretty = cli.pretty && texts.len() == 1;
            run_extract_dispatch(
                &engines,
                &processor,
                &texts,
                &schema,
                &meta,
                &opts,
                |offset, batch| {
                    if let Some((_, _, _, ref info)) = ls_state {
                        for (i, result) in batch.iter().enumerate() {
                            ls_tasks.push(labelstudio::convert_result_to_task(
                                &records[offset + i].text,
                                result,
                                info,
                            ));
                        }
                    }
                    write_jsonl_batch(
                        &mut out_writer,
                        &records[offset..offset + batch.len()],
                        &batch,
                        use_pretty,
                        &cli,
                    )
                },
            )?;
        }
    }

    // Upload accumulated tasks to Label Studio
    if let Some((ref url, ref api_key, project_id, _)) = ls_state {
        if !ls_tasks.is_empty() {
            labelstudio::import_tasks(url, api_key, project_id, &ls_tasks)?;
        }
    }

    Ok(())
}

fn create_engines(
    cli: &Cli,
    backend: &str,
    files: &ModelFiles,
    config: &ExtractorConfig,
    vocab: usize,
    num_workers: usize,
) -> Result<Vec<Engine>> {
    let mut engines = Vec::with_capacity(num_workers);
    for worker_idx in 0..num_workers {
        let engine = if backend == "tch" {
            #[cfg(feature = "tch")]
            {
                // For "cuda" without index, assign cuda:0, cuda:1, etc. per worker.
                // For "cuda:N" with index, assign cuda:N, cuda:N+1, etc.
                // For other devices (cpu, mps, vulkan), all workers share the same device.
                let device_str = if cli.device == "cuda" {
                    format!("cuda:{}", worker_idx)
                } else if cli.device.contains(':') {
                    let (base, idx) = cli.device.rsplit_once(':').unwrap();
                    let device_idx: usize = idx.parse().context("parse device index")?;
                    format!("{}:{}", base, device_idx + worker_idx)
                } else {
                    cli.device.clone()
                };
                let dev = parse_tch_device(&device_str)?;
                Engine::Tch(TchExtractor::load(files, config.clone(), vocab, dev)?)
            }
            #[cfg(not(feature = "tch"))]
            {
                let _ = worker_idx;
                anyhow::bail!("Backend \"tch\" requires building gliner2 with --features tch");
            }
        } else {
            #[cfg(feature = "candle")]
            {
                let _ = cli;
                Engine::Candle(CandleExtractor::load_cpu(files, config.clone(), vocab)?)
            }
            #[cfg(not(feature = "candle"))]
            {
                anyhow::bail!("Backend \"candle\" requires the default `candle` feature");
            }
        };
        engines.push(engine);
    }
    Ok(engines)
}

fn run_extract_streaming<F>(
    engine: &Engine,
    processor: &SchemaTransformer,
    texts: &[String],
    schema: &Value,
    meta: &gliner2::schema::ExtractionMetadata,
    opts: &ExtractOptions,
    on_batch: F,
) -> Result<()>
where
    F: FnMut(usize, Vec<Value>) -> Result<()>,
{
    match engine {
        #[cfg(feature = "candle")]
        Engine::Candle(e) => batch_extract_streaming(
            e,
            processor,
            texts,
            BatchSchemaMode::Shared { schema, meta },
            opts,
            on_batch,
        ),
        #[cfg(feature = "tch")]
        Engine::Tch(e) => batch_extract_streaming(
            e,
            processor,
            texts,
            BatchSchemaMode::Shared { schema, meta },
            opts,
            on_batch,
        ),
        #[cfg(not(any(feature = "candle", feature = "tch")))]
        _ => anyhow::bail!("No backend features (candle or tch) are enabled."),
    }
}

/// Dispatch to single-engine streaming when only one engine is available,
/// or to multi-engine parallel extraction otherwise.
fn run_extract_dispatch<F>(
    engines: &[Engine],
    processor: &SchemaTransformer,
    texts: &[String],
    schema: &Value,
    meta: &gliner2::schema::ExtractionMetadata,
    opts: &ExtractOptions,
    on_batch: F,
) -> Result<()>
where
    F: FnMut(usize, Vec<Value>) -> Result<()>,
{
    if engines.len() <= 1 {
        run_extract_streaming(&engines[0], processor, texts, schema, meta, opts, on_batch)
    } else {
        run_extract_streaming_multi(engines, processor, texts, schema, meta, opts, on_batch)
    }
}

fn run_extract_streaming_multi<F>(
    engines: &[Engine],
    processor: &SchemaTransformer,
    texts: &[String],
    schema: &Value,
    meta: &gliner2::schema::ExtractionMetadata,
    opts: &ExtractOptions,
    mut on_batch: F,
) -> Result<()>
where
    F: FnMut(usize, Vec<Value>) -> Result<()>,
{
    use rayon::prelude::*;

    if texts.is_empty() {
        return Ok(());
    }

    let num_workers = engines.len().max(1);
    let bs = opts.batch_size.max(1);

    // Collect all results with their original indices
    let mut all_results: Vec<(usize, Value)> = Vec::with_capacity(texts.len());

    // Process in chunks of batch_size * num_workers for better parallelism
    let chunk_size = bs * num_workers;
    let mut base = 0usize;

    for chunk in texts.chunks(chunk_size) {
        // Split chunk among workers — each gets up to `bs` texts
        let worker_results: Vec<Result<Vec<(usize, Value)>>> = (0..num_workers)
            .into_par_iter()
            .map(|worker_idx| {
                let start = worker_idx * bs;
                let end = (start + bs).min(chunk.len());
                if start >= chunk.len() {
                    return Ok(Vec::new());
                }

                let worker_texts = &chunk[start..end];
                let engine = &engines[worker_idx];
                let mut local_results = Vec::with_capacity(worker_texts.len());

                match engine {
                    #[cfg(feature = "candle")]
                    Engine::Candle(e) => {
                        batch_extract_streaming(
                            e,
                            processor,
                            worker_texts,
                            BatchSchemaMode::Shared { schema, meta },
                            opts,
                            |offset, batch| {
                                for (i, val) in batch.into_iter().enumerate() {
                                    local_results.push((start + offset + i, val));
                                }
                                Ok(())
                            },
                        )?;
                    }
                    #[cfg(feature = "tch")]
                    Engine::Tch(e) => {
                        batch_extract_streaming(
                            e,
                            processor,
                            worker_texts,
                            BatchSchemaMode::Shared { schema, meta },
                            opts,
                            |offset, batch| {
                                for (i, val) in batch.into_iter().enumerate() {
                                    local_results.push((start + offset + i, val));
                                }
                                Ok(())
                            },
                        )?;
                    }
                    #[cfg(not(any(feature = "candle", feature = "tch")))]
                    _ => anyhow::bail!("No backend features (candle or tch) are enabled."),
                }

                Ok(local_results)
            })
            .collect();

        // Merge results and sort by chunk-relative index to preserve input order
        for result in worker_results {
            let mut r = result?;
            all_results.append(&mut r);
        }
        all_results.sort_by_key(|(idx, _)| *idx);

        // Emit the sorted chunk as a single batch — base is the global offset into texts
        if !all_results.is_empty() {
            let batch: Vec<Value> = all_results.iter().map(|(_, val)| val.clone()).collect();
            on_batch(base, batch)?;
            base += all_results.len();
        }

        all_results.clear();
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Output writers (streaming)
// ---------------------------------------------------------------------------

fn create_jsonl_writer(cli: &Cli) -> Result<Box<dyn io::Write>> {
    if let Some(path) = &cli.output {
        Ok(Box::new(io::BufWriter::new(fs::File::create(path)?)))
    } else {
        Ok(Box::new(io::BufWriter::new(io::stdout().lock())))
    }
}

fn write_jsonl_batch(
    writer: &mut Box<dyn io::Write>,
    records: &[Record],
    results: &[Value],
    pretty: bool,
    cli: &Cli,
) -> Result<()> {
    for (i, r) in results.iter().enumerate() {
        let mut out_obj = serde_json::Map::new();
        if let Some(id) = &records[i].id {
            out_obj.insert(cli.id_field.clone(), id.clone());
        }
        out_obj.insert(cli.text_field.clone(), json!(records[i].text));
        out_obj.insert("result".into(), r.clone());
        if pretty {
            serde_json::to_writer_pretty(&mut *writer, &out_obj)?;
        } else {
            serde_json::to_writer(&mut *writer, &out_obj)?;
        }
        writeln!(writer)?;
    }
    writer.flush()?;
    Ok(())
}

fn create_parquet_writer(path: &Path, cli: &Cli) -> Result<parquet::arrow::ArrowWriter<fs::File>> {
    use arrow_schema::{DataType, Field, Schema};
    use parquet::basic::Compression;
    use parquet::file::properties::WriterProperties;
    use std::sync::Arc;

    let schema = Arc::new(Schema::new(vec![
        Field::new(&cli.id_field, DataType::Utf8, true),
        Field::new(&cli.text_field, DataType::Utf8, false),
        Field::new("result", DataType::Utf8, false),
    ]));

    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(Default::default()))
        .build();

    let file = fs::File::create(path)?;
    Ok(parquet::arrow::ArrowWriter::try_new(
        file,
        schema,
        Some(props),
    )?)
}

fn write_parquet_batch(
    writer: &mut parquet::arrow::ArrowWriter<fs::File>,
    records: &[Record],
    results: &[Value],
    cli: &Cli,
) -> Result<()> {
    use arrow_array::{RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;

    let schema = Arc::new(Schema::new(vec![
        Field::new(&cli.id_field, DataType::Utf8, true),
        Field::new(&cli.text_field, DataType::Utf8, false),
        Field::new("result", DataType::Utf8, false),
    ]));

    let ids: StringArray = records
        .iter()
        .map(|r| {
            r.id.as_ref().map(|v| match v {
                Value::String(s) => s.clone(),
                other => other.to_string(),
            })
        })
        .collect();

    let texts: StringArray = records.iter().map(|r| Some(r.text.as_str())).collect();

    let result_strings: StringArray = results
        .iter()
        .map(|r| Some(serde_json::to_string(r).unwrap_or_default()))
        .collect();

    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(ids), Arc::new(texts), Arc::new(result_strings)],
    )?;

    writer.write(&batch)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Input readers
// ---------------------------------------------------------------------------

fn read_jsonl(reader: impl BufRead, cli: &Cli) -> Result<Vec<Record>> {
    let mut records = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let val: Value = serde_json::from_str(&line)?;
        records.push(val_to_record(&val, cli)?);
    }
    Ok(records)
}

fn val_to_record(v: &Value, cli: &Cli) -> Result<Record> {
    let obj = v.as_object().context("Expected JSON object for record")?;
    let text = obj
        .get(&cli.text_field)
        .and_then(|t| t.as_str())
        .context(format!("Missing text field {:?} in record", cli.text_field))?
        .to_string();
    let id = obj.get(&cli.id_field).cloned();
    Ok(Record { id, text })
}

fn read_parquet_records(path: &Path, cli: &Cli) -> Result<Vec<Record>> {
    use arrow_array::Array;

    let file = fs::File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.build()?;

    let mut records = Vec::new();
    for batch in reader {
        let batch = batch?;
        let schema = batch.schema();
        let text_idx = schema.index_of(&cli.text_field).map_err(|_| {
            anyhow::anyhow!("Column {:?} not found in parquet file", cli.text_field)
        })?;
        let id_idx = schema.index_of(&cli.id_field).ok();

        let text_col = batch
            .column(text_idx)
            .as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .context(format!(
                "Column {:?} is not a string column",
                cli.text_field
            ))?;

        let id_col = id_idx.map(|idx| batch.column(idx));

        for row in 0..batch.num_rows() {
            let text = text_col.value(row).to_string();
            let id = id_col.and_then(|col| {
                let any = col.as_any();
                any.downcast_ref::<arrow_array::StringArray>()
                    .map(|s| json!(s.value(row)))
                    .or_else(|| {
                        any.downcast_ref::<arrow_array::Int64Array>()
                            .map(|i| json!(i.value(row)))
                    })
                    .or_else(|| {
                        any.downcast_ref::<arrow_array::Int32Array>()
                            .map(|i| json!(i.value(row)))
                    })
            });
            records.push(Record { id, text });
        }
    }
    Ok(records)
}

// ---------------------------------------------------------------------------
// Schema building
// ---------------------------------------------------------------------------

fn build_schema_and_meta(cmd: &Commands) -> Result<(Value, gliner2::schema::ExtractionMetadata)> {
    let mut s = gliner2::schema::create_schema();
    match cmd {
        Commands::Entities {
            label, labels_json, ..
        } => {
            if !label.is_empty() && labels_json.is_some() {
                anyhow::bail!("Cannot provide both --label and --labels-json");
            }
            if let Some(path) = labels_json {
                let v: Value = serde_json::from_str(&fs::read_to_string(path)?)?;
                s.entities(v);
            } else {
                s.entities(json!(label));
            }
        }
        Commands::Classify {
            task,
            label,
            labels_json,
            multi_label,
            cls_threshold,
            ..
        } => {
            if !label.is_empty() && labels_json.is_some() {
                anyhow::bail!("Cannot provide both --label and --labels-json");
            }
            let labels = if let Some(path) = labels_json {
                serde_json::from_str(&fs::read_to_string(path)?)?
            } else {
                json!(label)
            };
            s.classification(task, labels, *multi_label, *cls_threshold);
        }
        Commands::Relations {
            relation,
            relations_json,
            ..
        } => {
            if !relation.is_empty() && relations_json.is_some() {
                anyhow::bail!("Cannot provide both --relation and --relations-json");
            }
            let rels = if let Some(path) = relations_json {
                serde_json::from_str(&fs::read_to_string(path)?)?
            } else {
                json!(relation)
            };
            s.relations(rels);
        }
        Commands::Json {
            structures,
            structures_json,
            ..
        } => {
            if structures.is_some() && structures_json.is_some() {
                anyhow::bail!("Cannot provide both --structures and --structures-json");
            }
            if let Some(path) = structures {
                let v: Value = serde_json::from_str(&fs::read_to_string(path)?)?;
                let obj = v
                    .as_object()
                    .context("--structures must be a JSON object")?;
                s.extract_json_structures(obj)?;
            } else if let Some(js) = structures_json {
                let v: Value = serde_json::from_str(js)?;
                let obj = v
                    .as_object()
                    .context("--structures-json must be a JSON object")?;
                s.extract_json_structures(obj)?;
            }
        }
        Commands::Run { schema_file, .. } => {
            let v: Value = serde_json::from_str(&fs::read_to_string(schema_file)?)?;
            let meta = infer_metadata_from_schema(&v);
            return Ok((v, meta));
        }
        Commands::Setup { .. } | Commands::Status => unreachable!(),
    }
    Ok(s.build())
}
