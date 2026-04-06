use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use gliner2::config::{download_model, ModelFiles};
use gliner2::{
    batch_extract, infer_metadata_from_schema, BatchSchemaMode, ExtractOptions, ExtractorConfig,
    SchemaTransformer,
};
#[cfg(feature = "candle")]
use gliner2::CandleExtractor;
#[cfg(feature = "tch")]
use gliner2::TchExtractor;
use serde_json::{json, Value};
use std::fs;
use std::io::{self, BufRead};
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(name = "gliner2")]
#[command(version)]
#[command(about = "Gliner2 CLI", long_about = None)]
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
    #[arg(long, env = "GLINER2_BACKEND", default_value = "candle", global = true)]
    backend: String,

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
}

#[derive(Subcommand)]
enum Commands {
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

struct Record {
    id: Option<Value>,
    text: String,
}

enum Engine {
    #[cfg(feature = "candle")]
    Candle(CandleExtractor),
    #[cfg(feature = "tch")]
    Tch(TchExtractor),
}

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

fn main() -> Result<()> {
    let cli = Cli::parse();

    let level = match cli.log_level.to_lowercase().as_str() {
        "off" => tracing::Level::ERROR, // close enough if off isn't avail
        "error" => tracing::Level::ERROR,
        "warn" => tracing::Level::WARN,
        "info" => tracing::Level::INFO,
        "debug" => tracing::Level::DEBUG,
        "trace" => tracing::Level::TRACE,
        _ => tracing::Level::INFO,
    };
    tracing_subscriber::fmt().with_max_level(level).init();

    let files = resolve_model_files(&cli)?;

    let config: ExtractorConfig = serde_json::from_str(&fs::read_to_string(&files.config)?)?;

    let processor = SchemaTransformer::new(files.tokenizer.to_str().unwrap())?;
    let vocab = processor.tokenizer.get_vocab_size(true);

    let engine = if cli.backend == "tch" {
        #[cfg(feature = "tch")]
        {
            Engine::Tch(TchExtractor::load_cpu(&files, config, vocab)?)
        }
        #[cfg(not(feature = "tch"))]
        {
            anyhow::bail!("Backend \"tch\" requires building gliner2 with --features tch");
        }
    } else {
        #[cfg(feature = "candle")]
        {
            Engine::Candle(CandleExtractor::load_cpu(&files, config, vocab)?)
        }
        #[cfg(not(feature = "candle"))]
        {
            anyhow::bail!("Backend \"candle\" requires the default `candle` feature");
        }
    };

    let input_path = match &cli.command {
        Commands::Entities { input, .. } => input,
        Commands::Classify { input, .. } => input,
        Commands::Relations { input, .. } => input,
        Commands::Json { input, .. } => input,
        Commands::Run { input, .. } => input,
    };

    let records = gather_records(input_path, &cli)?;

    if records.is_empty() {
        return Ok(());
    }

    let (schema, meta) = build_schema_and_meta(&cli.command)?;

    let opts = ExtractOptions {
        threshold: cli.threshold,
        format_results: !cli.raw && cli.format_results,
        include_confidence: cli.include_confidence,
        include_spans: cli.include_spans,
        max_len: cli.max_len,
        batch_size: cli.batch_size,
    };

    let texts: Vec<String> = records.iter().map(|r| r.text.clone()).collect();
    let results = match &engine {
        #[cfg(feature = "candle")]
        Engine::Candle(e) => batch_extract(
            e,
            &processor,
            &texts,
            BatchSchemaMode::Shared {
                schema: &schema,
                meta: &meta,
            },
            &opts,
        )?,
        #[cfg(feature = "tch")]
        Engine::Tch(e) => batch_extract(
            e,
            &processor,
            &texts,
            BatchSchemaMode::Shared {
                schema: &schema,
                meta: &meta,
            },
            &opts,
        )?,
    };

    let mut out_writer: Box<dyn io::Write> = if let Some(path) = &cli.output {
        Box::new(fs::File::create(path)?)
    } else {
        Box::new(io::stdout())
    };

    if cli.pretty && results.len() == 1 {
        let r = &results[0];
        let mut out_obj = serde_json::Map::new();
        if let Some(id) = &records[0].id {
            out_obj.insert(cli.id_field.clone(), id.clone());
        }
        out_obj.insert(cli.text_field.clone(), json!(records[0].text));
        out_obj.insert("result".into(), r.clone());
        serde_json::to_writer_pretty(&mut out_writer, &out_obj)?;
        writeln!(out_writer)?;
    } else {
        for (i, r) in results.into_iter().enumerate() {
            let mut out_obj = serde_json::Map::new();
            if let Some(id) = &records[i].id {
                out_obj.insert(cli.id_field.clone(), id.clone());
            }
            out_obj.insert(cli.text_field.clone(), json!(records[i].text));
            out_obj.insert("result".into(), r);
            serde_json::to_writer(&mut out_writer, &out_obj)?;
            writeln!(out_writer)?;
        }
    }

    Ok(())
}

fn gather_records(input: &str, cli: &Cli) -> Result<Vec<Record>> {
    let mut records = Vec::new();
    let (mut reader, path_is_jsonl, path_is_json) = if input == "-" {
        (Box::new(io::BufReader::new(io::stdin())) as Box<dyn BufRead>, true, false)
    } else {
        let path = Path::new(input);
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        let is_jsonl = ext == "jsonl";
        let is_json = ext == "json";
        let file = fs::File::open(path)?;
        (Box::new(io::BufReader::new(file)) as Box<dyn BufRead>, is_jsonl, is_json)
    };

    if path_is_json {
        let val: Value = serde_json::from_reader(reader)?;
        if let Some(arr) = val.as_array() {
            for v in arr {
                records.push(val_to_record(v, cli)?);
            }
        } else {
            records.push(val_to_record(&val, cli)?);
        }
    } else if path_is_jsonl {
        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() { continue; }
            let val: Value = serde_json::from_str(&line)?;
            records.push(val_to_record(&val, cli)?);
        }
    } else {
        // Plain text
        if cli.text_split == "line" {
            for line in reader.lines() {
                let line = line?;
                if line.trim().is_empty() { continue; }
                records.push(Record { id: None, text: line });
            }
        } else {
            let mut content = String::new();
            reader.read_to_string(&mut content)?;
            records.push(Record { id: None, text: content });
        }
    }

    Ok(records)
}

fn val_to_record(v: &Value, cli: &Cli) -> Result<Record> {
    let obj = v.as_object().context("Expected JSON object for record")?;
    let text = obj.get(&cli.text_field)
        .and_then(|t| t.as_str())
        .context(format!("Missing text field {:?} in record", cli.text_field))?
        .to_string();
    let id = obj.get(&cli.id_field).cloned();
    Ok(Record { id, text })
}

fn build_schema_and_meta(cmd: &Commands) -> Result<(Value, gliner2::schema::ExtractionMetadata)> {
    let mut s = gliner2::schema::create_schema();
    match cmd {
        Commands::Entities { label, labels_json, .. } => {
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
        Commands::Classify { task, label, labels_json, multi_label, cls_threshold, .. } => {
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
        Commands::Relations { relation, relations_json, .. } => {
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
        Commands::Json { structures, structures_json, .. } => {
            if structures.is_some() && structures_json.is_some() {
                anyhow::bail!("Cannot provide both --structures and --structures-json");
            }
            if let Some(path) = structures {
                let v: Value = serde_json::from_str(&fs::read_to_string(path)?)?;
                let obj = v.as_object().context("--structures must be a JSON object")?;
                s.extract_json_structures(obj)?;
            } else if let Some(js) = structures_json {
                let v: Value = serde_json::from_str(js)?;
                let obj = v.as_object().context("--structures-json must be a JSON object")?;
                s.extract_json_structures(obj)?;
            }
        }
        Commands::Run { schema_file, .. } => {
            let v: Value = serde_json::from_str(&fs::read_to_string(schema_file)?)?;
            let meta = infer_metadata_from_schema(&v);
            return Ok((v, meta));
        }
    }
    Ok(s.build())
}
