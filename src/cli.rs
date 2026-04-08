#[cfg(feature = "candle")]
use crate::CandleExtractor;
use crate::config::{ModelFiles, download_model};
use crate::{
    BatchSchemaMode, ExtractOptions, ExtractorConfig, SchemaTransformer, batch_extract,
    infer_metadata_from_schema,
};
#[cfg(feature = "tch")]
use crate::{TchExtractor, parse_tch_device};
use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde_json::{Value, json};
use std::fs;
use std::io::{self, BufRead};
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(name = "gliner2")]
#[command(version)]
#[command(about = "Gliner2 CLI", long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,

    /// Hugging Face model id
    #[arg(long, default_value = "fastino/gliner2-base-v1", global = true)]
    pub model: String,

    /// Offline layout directory
    #[arg(long, global = true)]
    pub model_dir: Option<PathBuf>,

    /// Explicit path to config.json
    #[arg(long, global = true)]
    pub config: Option<PathBuf>,

    /// Explicit path to encoder_config/config.json
    #[arg(long, global = true)]
    pub encoder_config: Option<PathBuf>,

    /// Explicit path to tokenizer.json
    #[arg(long, global = true)]
    pub tokenizer: Option<PathBuf>,

    /// Explicit path to model.safetensors
    #[arg(long, global = true)]
    pub weights: Option<PathBuf>,

    /// Backend (candle or tch)
    #[arg(long, env = "GLINER2_BACKEND", global = true)]
    pub backend: Option<String>,

    /// LibTorch device when using `--backend tch` (ignored for candle): cpu, cuda, cuda:N, mps, vulkan, auto.
    #[arg(long, env = "GLINER2_DEVICE", default_value = "cpu", global = true)]
    pub device: String,

    /// Log level (off, error, warn, info, debug, trace)
    #[arg(long, default_value = "info", global = true)]
    pub log_level: String,

    // Inference flags
    #[arg(long, default_value_t = 0.5, global = true)]
    pub threshold: f32,

    #[arg(long, global = true)]
    pub max_len: Option<usize>,

    #[arg(long, global = true)]
    pub include_confidence: bool,

    #[arg(long, global = true)]
    pub include_spans: bool,

    #[arg(long, default_value_t = true, action = clap::ArgAction::Set, global = true)]
    pub format_results: bool,

    #[arg(long, global = true)]
    pub raw: bool,

    #[arg(long, default_value_t = 8, global = true)]
    pub batch_size: usize,

    /// Field containing document text in JSON / JSONL records
    #[arg(long, default_value = "text", global = true)]
    pub text_field: String,

    /// Field to pass through as record id when present
    #[arg(long, default_value = "id", global = true)]
    pub id_field: String,

    /// Plain text: full (whole file) or line (one record per non-empty line)
    #[arg(long, default_value = "full", global = true)]
    pub text_split: String,

    /// Output path (default: stdout)
    #[arg(short, long, global = true)]
    pub output: Option<PathBuf>,

    /// Pretty-print JSON (only if output can be buffered)
    #[arg(long, global = true)]
    pub pretty: bool,
}

#[derive(Subcommand)]
pub enum Commands {
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

/// Run inference with the given parsed CLI arguments and resolved backend name.
pub fn run(cli: Cli, backend: &str) -> Result<()> {
    let files = resolve_model_files(&cli)?;

    let config: ExtractorConfig = serde_json::from_str(&fs::read_to_string(&files.config)?)?;

    let processor = SchemaTransformer::new(files.tokenizer.to_str().unwrap())?;
    let vocab = processor.tokenizer.get_vocab_size(true);

    let engine = if backend == "tch" {
        #[cfg(feature = "tch")]
        {
            let dev = parse_tch_device(&cli.device)?;
            Engine::Tch(TchExtractor::load(&files, config, vocab, dev)?)
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

    let input = Input::parse(input_path)?;
    let output_format = Output::resolve(&input, &cli);

    let (schema, meta) = build_schema_and_meta(&cli.command)?;

    let opts = ExtractOptions {
        threshold: cli.threshold,
        format_results: !cli.raw && cli.format_results,
        include_confidence: cli.include_confidence,
        include_spans: cli.include_spans,
        max_len: cli.max_len,
        batch_size: cli.batch_size,
    };

    // Multi-file parquet: process each file independently
    if let Input::ParquetGlob(ref paths) = input {
        if let Output::Parquet(ref out_dir) = output_format {
            fs::create_dir_all(out_dir)?;
            for in_path in paths {
                let records = read_parquet_records(in_path, &cli)?;
                if records.is_empty() {
                    continue;
                }
                let texts: Vec<String> = records.iter().map(|r| r.text.clone()).collect();
                let results = run_extract(&engine, &processor, &texts, &schema, &meta, &opts)?;
                let out_path =
                    out_dir.join(in_path.file_name().context("input file has no filename")?);
                write_parquet_output(&out_path, &records, &results, &cli)?;
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
    let results = run_extract(&engine, &processor, &texts, &schema, &meta, &opts)?;

    match output_format {
        Output::Parquet(ref path) => {
            write_parquet_output(path, &records, &results, &cli)?;
        }
        Output::Jsonl => {
            write_jsonl_output(&records, &results, &cli)?;
        }
    }

    Ok(())
}

fn run_extract(
    engine: &Engine,
    processor: &SchemaTransformer,
    texts: &[String],
    schema: &Value,
    meta: &crate::schema::ExtractionMetadata,
    opts: &ExtractOptions,
) -> Result<Vec<Value>> {
    match engine {
        #[cfg(feature = "candle")]
        Engine::Candle(e) => batch_extract(
            e,
            processor,
            texts,
            BatchSchemaMode::Shared { schema, meta },
            opts,
        ),
        #[cfg(feature = "tch")]
        Engine::Tch(e) => batch_extract(
            e,
            processor,
            texts,
            BatchSchemaMode::Shared { schema, meta },
            opts,
        ),
    }
}

fn write_jsonl_output(records: &[Record], results: &[Value], cli: &Cli) -> Result<()> {
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
        for (i, r) in results.iter().enumerate() {
            let mut out_obj = serde_json::Map::new();
            if let Some(id) = &records[i].id {
                out_obj.insert(cli.id_field.clone(), id.clone());
            }
            out_obj.insert(cli.text_field.clone(), json!(records[i].text));
            out_obj.insert("result".into(), r.clone());
            serde_json::to_writer(&mut out_writer, &out_obj)?;
            writeln!(out_writer)?;
        }
    }

    Ok(())
}

const PARQUET_BATCH_SIZE: usize = 8192;

fn write_parquet_output(
    path: &Path,
    records: &[Record],
    results: &[Value],
    cli: &Cli,
) -> Result<()> {
    use arrow_array::{RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use parquet::arrow::ArrowWriter;
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
    let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;

    for chunk_start in (0..records.len()).step_by(PARQUET_BATCH_SIZE) {
        let chunk_end = (chunk_start + PARQUET_BATCH_SIZE).min(records.len());
        let chunk_records = &records[chunk_start..chunk_end];
        let chunk_results = &results[chunk_start..chunk_end];

        let ids: StringArray = chunk_records
            .iter()
            .map(|r| {
                r.id.as_ref().map(|v| match v {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                })
            })
            .collect();

        let texts: StringArray = chunk_records
            .iter()
            .map(|r| Some(r.text.as_str()))
            .collect();

        let result_strings: StringArray = chunk_results
            .iter()
            .map(|r| Some(serde_json::to_string(r).unwrap_or_default()))
            .collect();

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(ids), Arc::new(texts), Arc::new(result_strings)],
        )?;

        writer.write(&batch)?;
    }

    writer.close()?;
    Ok(())
}

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

fn build_schema_and_meta(cmd: &Commands) -> Result<(Value, crate::schema::ExtractionMetadata)> {
    let mut s = crate::schema::create_schema();
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
    }
    Ok(s.build())
}
