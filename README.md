# Gliner2 Rust

[![Latest version](https://img.shields.io/crates/v/gliner2.svg)](https://crates.io/crates/gliner2)
[![Documentation](https://docs.rs/gliner2/badge.svg)](https://docs.rs/gliner2)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square)](https://github.com/huggingface/gliner2/blob/main/LICENSE)

This project implements the [Gliner2](https://github.com/fastino-ai/GLiNER2) model in rust with compatibility to the original weights and output of the python training.

```bash
cargo add gliner2
# and/or for a cli utility
cargo install gliner2
```

## Recorded speed (comparison harness)

The [harness/](harness/) scripts run the same **release** Rust binaries (`harness_compare`, `harness_compare_mt` on CPU) against the PyPI `gliner2` package. Timing fields are wall-clock milliseconds from a single process: `load_model_ms` is one-time load; `infer_ms` is per-fixture forward work (entity harness sums all cases for the total row).

**Reproduce (CPU vs CPU):** from the repo root, with Hugging Face access for the default model:

```bash
uv sync --locked --directory harness
bash harness/run_all.sh
bash harness/run_multitask.sh
```

The shell wrappers call Python with `CUDA_VISIBLE_DEVICES=` and `--device cpu` so PyTorch does not use a discrete NVIDIA GPU and weights stay on CPU, matching the Rust side.

For **apples-to-apples timing** with the Rust single-forward path, Python uses `**batch_size=1`**: `batch_extract_entities([text], …, batch_size=1)` on the entity harness and `batch_extract([text], schema, batch_size=1, …)` on the multitask harness (instead of relying on `extract` / `extract_entities` defaults).

**Reading `python/rust`:** for infer times this is `(python infer_ms) / (rust infer_ms)` per case or for the total line. Values **below 1** mean Python spent less time on that measure for these fixtures; **above 1** mean Python was slower.

### CPU vs CPU (recorded)

Model: `fastino/gliner2-base-v1`. **Recorded:** 2026-04-05 (Linux x86_64, local run; numbers vary by machine and load).

**Entity harness** ([harness/fixtures.json](harness/fixtures.json)) — metadata and per-case infer times:


|                              | Rust  | Python     |
| ---------------------------- | ----- | ---------- |
| `device_note`                | `cpu` | `cpu`      |
| `load_model_ms`              | 445.0 | 3874.4     |
| Sum of `infer_ms` over cases | 569.9 | 358.7      |
| `python/rust` (total infer)  | —     | **0.629×** |



| Case id             | rust `infer_ms` | python `infer_ms` | `python/rust` |
| ------------------- | --------------- | ----------------- | ------------- |
| `steve_jobs`        | 140.0           | 118.9             | 0.849×        |
| `tim_cook_iphone`   | 151.1           | 85.6              | 0.566×        |
| `sundar_pichai`     | 144.3           | 78.8              | 0.546×        |
| `microsoft_windows` | 134.4           | 75.4              | 0.561×        |


**Multitask harness** ([harness/fixtures_multitask.json](harness/fixtures_multitask.json)) — single fixture `entities_plus_sentiment`:


|                             | Rust  | Python     |
| --------------------------- | ----- | ---------- |
| `device_note`               | `cpu` | `cpu`      |
| `load_model_ms`             | 409.5 | 4002.4     |
| Sum of `infer_ms`           | 157.5 | 113.9      |
| `python/rust` (total infer) | —     | **0.724×** |


These are **short-fixture** timings. Update the tables when you change the model, fixtures, or harness code in a way that affects performance.

### Throughput (local only; not in CI)

**These benchmarks are not run in GitHub Actions** (see [.github/workflows/ci.yml](.github/workflows/ci.yml)). Run them on your machine when you need larger-sample timing.

The harness uses **64 samples** by default, built by cycling texts from [harness/fixtures.json](harness/fixtures.json). Every sample uses the same entity label list `["company", "person", "product", "location", "date"]` so Rust [batch_extract_entities](src/extract.rs) and PyPI `batch_extract_entities` can process the full set with `**batch_size=64`**. **Sequential rows** use **64× micro-batches of size 1** on both sides (Rust’s `forward` loop vs Python `batch_extract_entities([t], …, batch_size=1)`). **Batched rows** use one logical batch of 64 on each side.

```bash
uv sync --locked --directory harness
bash harness/run_throughput.sh
```

Optional: `bash harness/run_throughput.sh [fixtures.json] [rust_seq_out.json] [rust_batch_out.json] [samples]`. The script runs [harness/compare_throughput.py](harness/compare_throughput.py) on the three JSON outputs.

**Recorded:** 2026-04-05 (Linux x86_64, CPU, `CUDA_VISIBLE_DEVICES=` + `--device cpu` on Python). `warmup_full_passes=2` over all samples before each timed pass.


| Lane                               | `total_infer_ms` (64 samples) | samples/s | `python/rust` (infer) |
| ---------------------------------- | ----------------------------- | --------- | --------------------- |
| Rust sequential (`batch_size` 1)   | 8813.2                        | 7.26      | —                     |
| Python sequential (`batch_size` 1) | 4843.0                        | 13.22     | **0.550×**            |
| Rust batched (`batch_size` 64)     | 6794.2                        | 9.42      | —                     |
| Python batched (`batch_size` 64)   | 1650.6                        | 38.78     | **0.243×**            |


Load times from that run: Rust sequential ~492 ms, Rust batched ~467 ms, Python ~2613 ms.

Re-run `bash harness/run_throughput.sh` to refresh; the script prints the same layout via [harness/compare_throughput.py](harness/compare_throughput.py).

### GPU vs GPU (not recorded yet)

Fair comparison needs **both** implementations on the same device class (for example CUDA on the PyPI side and a GPU inference path in the Rust harness). That pairing is not wired into the harness yet, so no GPU numbers are published here.


|                  | Rust | Python |
| ---------------- | ---- | ------ |
| Device           | —    | —      |
| `load_model_ms`  | —    | —      |
| Total `infer_ms` | —    | —      |
| `python/rust`    | —    | —      |


## Usage

Like the Python implementation, this crate supports a full extraction API. You load the model once, build a `SchemaTransformer` from the tokenizer, then call `Extractor` methods.

### Setup (load model + tokenizer)

```rust
use anyhow::Result;
use candle_core::Device;
use candle_nn::VarBuilder;
use candle_transformers::models::debertav2::Config as DebertaConfig;
use gliner2::config::{download_model, ExtractorConfig};
use gliner2::{Extractor, SchemaTransformer};

fn load_extractor(model_id: &str) -> Result<(Extractor, SchemaTransformer)> {
    let files = download_model(model_id)?;
    let device = Device::Cpu;
    let dtype = candle_core::DType::F32;

    let config: ExtractorConfig = serde_json::from_str(&std::fs::read_to_string(&files.config)?)?;
    let mut encoder_config: DebertaConfig =
        serde_json::from_str(&std::fs::read_to_string(&files.encoder_config)?)?;
    let transformer = SchemaTransformer::new(files.tokenizer.to_str().unwrap())?;
    encoder_config.vocab_size = transformer.tokenizer.get_vocab_size(true);

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[files.weights], dtype, &device)? };
    let extractor = Extractor::load(config, encoder_config, vb)?;
    Ok((extractor, transformer))
}
```

### Entity extraction (`extract_entities`)

Same idea as Python `extract_entities`: pass label names; the returned `serde_json::Value` uses the formatted shape (`entities` → label → list of strings, when `include_spans` / `include_confidence` are false).

```rust
use gliner2::ExtractOptions;
use serde_json::json;

let (extractor, transformer) = load_extractor("fastino/gliner2-base-v1")?;
let text = "Apple CEO Tim Cook announced iPhone 15 in Cupertino.";

let entity_types = vec![
    "company".to_string(),
    "person".to_string(),
    "product".to_string(),
    "location".to_string(),
];

let opts = ExtractOptions::default();
let out = extractor.extract_entities(&transformer, text, &entity_types, &opts)?;
// e.g. {"entities":{"company":["Apple"],"person":["Tim Cook"], ...}}

// Optional: character spans + confidence (richer JSON, closer to Python with flags on)
let opts_rich = ExtractOptions {
    include_confidence: true,
    include_spans: true,
    ..Default::default()
};
let _out = extractor.extract_entities(&transformer, text, &entity_types, &opts_rich)?;
```

### Text classification (`classify_text`)

One classification task per call. `labels` is a JSON array of class names, or an object mapping label → description (like Python).

```rust
use gliner2::ExtractOptions;
use serde_json::json;

let (extractor, transformer) = load_extractor("fastino/gliner2-base-v1")?;
let text = "The new phone is amazing and well worth the price.";

// Single-label: scalar string under the task name when format_results is true
let opts = ExtractOptions::default();
let out = extractor.classify_text(
    &transformer,
    text,
    "sentiment",
    json!(["positive", "negative", "neutral"]),
    &opts,
)?;
// e.g. {"sentiment":"positive"}

// Labels with optional descriptions (mirrors Python dict form)
let out2 = extractor.classify_text(
    &transformer,
    text,
    "topic",
    json!({
        "technology": "Tech products and software",
        "business": "Corporate or market news",
        "sports": "Athletics and games"
    }),
    &opts,
)?;
```

### Relation extraction (`extract_relations`)

Pass relation names as a JSON array of strings, or a JSON object (name → description / config), matching Python `relations(...)`.

```rust
use gliner2::ExtractOptions;
use serde_json::json;

let (extractor, transformer) = load_extractor("fastino/gliner2-base-v1")?;
let text = "Tim Cook works for Apple, based in Cupertino.";

let opts = ExtractOptions::default();

// List of relation types → formatted results under "relation_extraction"
let out = extractor.extract_relations(
    &transformer,
    text,
    json!(["works_for", "located_in"]),
    &opts,
)?;
// e.g. {"relation_extraction":{"works_for":[["Tim Cook","Apple"]],"located_in":[["Apple","Cupertino"]]}}

// Dict form (descriptions stored like Python; inference uses relation names)
let _out2 = extractor.extract_relations(
    &transformer,
    text,
    json!({
        "works_for": "Employment between person and organization",
        "founded": "Founder relationship"
    }),
    &opts,
)?;
```

### Structured JSON (`extract_json`)

Field specs use the same string syntax as Python `extract_json` (`name::dtype::[choices]::description`).

```rust
use gliner2::ExtractOptions;
use serde_json::json;

let (extractor, transformer) = load_extractor("fastino/gliner2-base-v1")?;
let text = "iPhone 15 Pro costs $999 and is in stock.";

let structures = json!({
    "product_info": [
        "name::str",
        "price::str",
        "features::list",
        "availability::str::[in_stock|pre_order|sold_out]"
    ]
});
let out = extractor.extract_json(
    &transformer,
    text,
    &structures,
    &ExtractOptions::default(),
)?;
```

### Multi-task builder (`create_schema` + `extract`)

Combines entities, classifications, relations, and structured fields in one encoder pass. Uses the same `(extractor, transformer)` and `text` as in the setup section.

```rust
use gliner2::{
    create_schema, ExtractOptions, Extractor, SchemaTransformer, ValueDtype,
};
use serde_json::json;

let mut s = create_schema();
s.entities(json!({
    "person": "Names of people",
    "company": "Organization names",
    "product": "Products or offerings",
}));
s.classification_simple("sentiment", json!(["positive", "negative", "neutral"]));
s.classification_simple("category", json!(["technology", "business", "finance", "healthcare"]));
s.relations(json!(["works_for", "founded", "located_in"]));
{
    let _ = s.structure("product_info")
        .field_str("name")
        .field_str("price")
        .field_list("features")
        .field_choices(
            "availability",
            vec![
                "in_stock".into(),
                "pre_order".into(),
                "sold_out".into(),
            ],
            ValueDtype::Str,
        );
}
let (schema_val, meta) = s.build();
let opts = ExtractOptions::default();
let out = extractor.extract(&transformer, text, &schema_val, &meta, &opts)?;
```

### Batch inference

The crate mirrors Python’s batched entry points: records are preprocessed, **padded into chunks** of at most `ExtractOptions::batch_size` (default **8**), the encoder runs **once per chunk**, span representations are computed with **`compute_span_rep_batched`** when needed, then each row is decoded. Results are returned in **input order**.

Set `batch_size` on `ExtractOptions` for any batch method (it only affects chunking, not single-sample `extract_*` calls).

#### Shared schema (one schema for every text)

Use the `Extractor` helpers; they build the same schema as the single-sample methods and call `batch_extract` internally.

```rust
use gliner2::ExtractOptions;
use serde_json::json;

let (extractor, transformer) = load_extractor("fastino/gliner2-base-v1")?;
let texts: Vec<String> = vec![
    "Apple CEO Tim Cook announced iPhone 15.".into(),
    "Google unveiled Gemini in Mountain View.".into(),
];

let entity_types: Vec<String> = ["company", "person", "product", "location"]
    .into_iter()
    .map(String::from)
    .collect();

let mut opts = ExtractOptions::default();
opts.batch_size = 16;

let results = extractor.batch_extract_entities(&transformer, &texts, &entity_types, &opts)?;
// Vec<serde_json::Value>, one formatted result per input line

let cls = extractor.batch_classify_text(
    &transformer,
    &texts,
    "sentiment",
    json!(["positive", "negative", "neutral"]),
    &opts,
)?;

let rels = extractor.batch_extract_relations(
    &transformer,
    &texts,
    json!(["works_for", "located_in"]),
    &opts,
)?;

let structures = json!({
    "product_info": ["name::str", "price::str"]
});
let json_results = extractor.batch_extract_json(&transformer, &texts, &structures, &opts)?;
```

#### Full schema + metadata (`batch_extract`)

For the same multitask flow as [`extract`](#multi-task-builder-create_schema--extract), build `(schema_val, meta)` once and run **`batch_extract`** with **`BatchSchemaMode::Shared`**, or pass per-row schemas and metadata with **`BatchSchemaMode::PerSample`**.

```rust
use gliner2::{batch_extract, create_schema, BatchSchemaMode, ExtractOptions};
use gliner2::schema::infer_metadata_from_schema;
use serde_json::{json, Value};

let (extractor, transformer) = load_extractor("fastino/gliner2-base-v1")?;
let texts: Vec<String> = vec!["First document.".into(), "Second document.".into()];

// Option A — shared multitask schema from the builder
let mut s = create_schema();
s.entities(json!({ "company": "", "person": "" }));
s.classification_simple("sentiment", json!(["positive", "negative", "neutral"]));
let (schema_val, meta) = s.build();

let opts = ExtractOptions {
    batch_size: 8,
    ..Default::default()
};

let out_shared = batch_extract(
    &extractor,
    &transformer,
    &texts,
    BatchSchemaMode::Shared {
        schema: &schema_val,
        meta: &meta,
    },
    &opts,
)?;

// Option B — per-text JSON schemas (e.g. from config); metadata from infer_metadata_from_schema
let schema_a: Value = json!({ "entities": { "person": "" } });
let schema_b: Value = json!({ "entities": { "location": "" } });
let schemas = vec![schema_a.clone(), schema_b.clone()];
let metas = vec![
    infer_metadata_from_schema(&schema_a),
    infer_metadata_from_schema(&schema_b),
];

let out_per = batch_extract(
    &extractor,
    &transformer,
    &texts,
    BatchSchemaMode::PerSample {
        schemas: &schemas,
        metas: &metas,
    },
    &opts,
)?;
```

For a shared schema you can also call **`extractor.batch_extract(&transformer, &texts, &schema_val, &meta, &opts)`** instead of the free function.

Lower-level reuse: after **`transform_extract`** you can run **`extract_from_preprocessed`** on one sample if you already have encoder outputs and span tensors; see [`src/extract.rs`](src/extract.rs).

## Development

### Pre-commit

Git hooks run the same Rust checks as CI (`cargo fmt`, `cargo clippy` on the workspace) plus [Ruff](https://docs.astral.sh/ruff/) on first-party Python (for example under `harness/`). Paths under `reference/` and `.tickets/` are excluded from hooks.

**Prerequisites:** stable Rust with `rustfmt` and `clippy` (for example `rustup component add rustfmt clippy`).

**Install** [pre-commit](https://pre-commit.com/) (either is fine):

```bash
uv tool install pre-commit
```

From the repository root, install the hooks once:

```bash
pre-commit install
```

Optionally validate the whole tree:

```bash
pre-commit run --all-files
```

If you must commit before fixing Clippy, you can skip that hook: `SKIP=cargo-clippy git commit` (use sparingly; CI still enforces warnings as errors).

## CLI specification

The command-line interface is **not implemented yet**. This section specifies the intended `gliner2` binary (see `default-run` in `Cargo.toml`) so future work can match the library API and Python `GLiNER2` behavior.

Install the binary with `cargo install gliner2`. Inference flags mirror [ExtractOptions](src/extract.rs) (`threshold`, `format_results`, `include_confidence`, `include_spans`, `max_len`).

### Command overview

```mermaid
flowchart LR
  subgraph sub [Subcommands]
    entities[entities]
    classify[classify]
    relations[relations]
    jsonCmd[json]
    run[run]
  end
  gliner2[gliner2] --> entities
  gliner2 --> classify
  gliner2 --> relations
  gliner2 --> jsonCmd
  gliner2 --> run
```




| Subcommand          | Purpose                                      | Library analogue                                             |
| ------------------- | -------------------------------------------- | ------------------------------------------------------------ |
| `gliner2 entities`  | Named-entity extraction                      | `Extractor::extract_entities`, `Schema::entities`            |
| `gliner2 classify`  | Text classification (single- or multi-label) | `Extractor::classify_text`, `Schema::classification`         |
| `gliner2 relations` | Relation extraction                          | `Extractor::extract_relations`, `Schema::relations`          |
| `gliner2 json`      | Structured JSON / field extraction           | `Extractor::extract_json`, `Schema::extract_json_structures` |
| `gliner2 run`       | Multitask: full engine schema in one pass    | `Extractor::extract`                                         |


Top-level: `gliner2 --help`, `gliner2 --version`, and `gliner2 <subcommand> --help`.

### Global options

These apply to every subcommand unless stated otherwise.


| Flag                                                       | Description                                                                                                                                                       |
| ---------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--model <HF_REPO_ID>`                                     | Hugging Face model id (default: `fastino/gliner2-base-v1`, same as `harness/` scripts).                                                                           |
| `--model-dir <DIR>`                                        | Offline layout: `config.json`, `encoder_config/config.json`, `tokenizer.json`, `model.safetensors` (matches `ModelFiles` from [download_model](src/config.rs)). |
| `--config`, `--encoder-config`, `--tokenizer`, `--weights` | Explicit paths instead of `--model` / `--model-dir`.                                                                                                              |
| `-q`, `-v` / `--log-level`                                 | Quiet / verbose logging (exact mapping is implementation-defined).                                                                                                |


Use either Hub resolution (`--model`) **or** a local layout (`--model-dir` or explicit file flags), not a conflicting mix; if both are given, the implementation should reject the invocation with a clear error.

**Device and dtype** are intentionally unspecified here until the library exposes them; do not document GPU flags until they exist.

### Shared inference flags


| Flag                            | Maps to                     | Default                   |
| ------------------------------- | --------------------------- | ------------------------- |
| `--threshold <float>`           | `ExtractOptions::threshold` | `0.5`                     |
| `--max-len <N>`                 | `ExtractOptions::max_len`   | unset                     |
| `--include-confidence`          | `include_confidence`        | off                       |
| `--include-spans`               | `include_spans`             | off                       |
| `--raw` / `--no-format-results` | `format_results = false`    | formatted output (`true`) |


### Batching

The **library** implements tensor batch inference (`Extractor::batch_extract*`, `ExtractOptions::batch_size`); see **[Batch inference](#batch-inference)** above. The **CLI** is not implemented yet; the contract below assumes the binary will drive those batched APIs for any input that produces **more than one logical record** (for example multi-line JSONL or plain text with `--text-split line` and multiple non-empty lines).


| Flag               | Description                                                                                                                                        |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--batch-size <N>` | Maximum records per model batch. Default: **8** (implementation may choose a lower value on constrained devices, but must document any deviation). |
| `--batch-size 1`   | Effectively sequential inference (debugging, peak memory limits, or until batched paths are stable).                                               |


**Single-record** inputs (one JSONL line, one JSON object, or `--text-split full` over an entire file) form a single batch of size 1.

**Ordering:** Output lines must follow **the same order as input records**, even when flushing internal batches.

### Input and output

**Input:** final positional argument `INPUT`, or `-` for stdin.


| Flag                  | Description                                                                                                                     |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `--text-field <KEY>`  | Field containing document text in JSON / JSONL records (default: `text`).                                                       |
| `--id-field <KEY>`    | Field to pass through as record id when present (default: `id`).                                                                |
| `--text-split <MODE>` | Plain text: `full` (whole file) or `line` (one record per non-empty line). `sentence` / `char-chunk` reserved. Default: `full`. |



| Format         | Detection / notes                                                                                                                                                                                                                                              |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **JSONL**      | One JSON object per line. Text from `--text-field` (default: `text`). If the input object contains the id key named by `--id-field` (default: `id`), copy that field through to the output object.                                                             |
| **JSON**       | A single object using the same field convention. For many records, use JSONL or preprocess (for example with `jq`).                                                                                                                                            |
| **Plain text** | Controlled by `--text-split`: `full` (default for `.txt`) — entire file is one record; `line` — each non-empty line is one record (multiple lines ⇒ batching). `**sentence` and `char-chunk`** are reserved for a future release (segmentation semantics TBD). |


**Output:** JSONL to stdout by default. `--output <PATH>` / `-o <PATH>` (use `-` for stdout). Optional `--pretty`: pretty-printed JSON when the implementation can buffer a single record or full result (for example one JSON object input or explicit single-line mode).

**Format inference:** From `INPUT`’s path suffix when possible: `.jsonl` → JSONL, `.json` → single JSON object, `.txt` (or other) → plain text with `--text-split`. For stdin (`-`), default input format is **JSONL** (one object per line).

### Output record shape

Each output line is one JSON object, for example:

```json
{"id":"optional","text":"...","result":{ }}
```

`result` matches Python / Rust `**format_results**` output for the task mix (entities, `relation_extraction`, classification keys, structured parents, etc.), consistent with the harness direction in `harness/compare.py` and multitask fixtures. If the input record has no `id`, omit `id` from the output (or use `null`; implementations should pick one behavior and document it).

### Subcommands

#### `gliner2 entities`


| Flag                   | Description                                                                                                                                 |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `--label <NAME>`       | Repeatable entity type name.                                                                                                                |
| `--labels-json <PATH>` | JSON array of names or object form accepted by `Schema::entities` (name → description string or `{ "description", "dtype", "threshold" }`). |


**Precedence:** If any `--label` is given **and** `--labels-json` is given, exit with a usage error (do not merge).

#### `gliner2 classify`


| Flag                      | Description                                                                     |
| ------------------------- | ------------------------------------------------------------------------------- |
| `--task <NAME>`           | Required classification task name (JSON key in formatted output).               |
| `--label <NAME>`          | Repeatable class label.                                                         |
| `--labels-json <PATH>`    | Array of labels or object label → description (Python-style).                   |
| `--multi-label`           | Multi-label classification (`Schema::classification` with `multi_label: true`). |
| `--cls-threshold <float>` | Per-task classifier threshold (default `0.5`).                                  |


Same rule: do not combine `--label` with `--labels-json`.

#### `gliner2 relations`


| Flag                      | Description                                                         |
| ------------------------- | ------------------------------------------------------------------- |
| `--relation <NAME>`       | Repeatable relation type name.                                      |
| `--relations-json <PATH>` | JSON array of names or object form accepted by `Schema::relations`. |


Do not pass both repeatable `--relation` and `--relations-json`.

#### `gliner2 json`


| Flag                           | Description                                                      |
| ------------------------------ | ---------------------------------------------------------------- |
| `--structures <PATH>`          | JSON file: object mapping structure name → array of field specs. |
| `--structures-json '<OBJECT>'` | Same object inline.                                              |


Field specs use the same grammar as **Structured JSON (`extract_json`)** above: strings like `name::dtype::[choices]::description` or JSON objects parsed by [parse_field_spec](src/schema.rs). Do not pass both `--structures` and `--structures-json`.

#### `gliner2 run`


| Flag                   | Description                                                                                                                                                                                                                                                                                                             |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--schema-file <PATH>` | Required. Full **engine** multitask schema (same shape as Python `GLiNER2.extract(text, schema)`). See [harness/fixtures_multitask.json](harness/fixtures_multitask.json) for a minimal example: `entities`, `classifications`, `relations`, `json_structures`, optional `entity_descriptions` / `json_descriptions`. |


Each entry in `classifications` should include `"true_label": ["N/A"]` when mirroring Python; the harness script [harness/run_multitask_python.py](harness/run_multitask_python.py) sets this if missing.

### Environment

- `**HF_TOKEN`** — access to private or gated Hub models.
- Cache and offline behavior follow [Hugging Face Hub](https://huggingface.co/docs/huggingface_hub/index) environment variables (`HF_HOME`, etc.); see upstream docs for the full list.

### Exit codes

- **0** — success.
- **Non-zero** — usage errors, I/O failures, model load failures, or inference errors.

### Examples

```bash
# Entities: JSONL in → JSONL out (multi-record; default --batch-size 8 unless overridden)
gliner2 entities --label company --label person --batch-size 16 docs.jsonl --output out.jsonl

# Classify with labels from a file (JSONL input)
gliner2 classify --task sentiment --labels-json labels.json tweets.jsonl

# Relations
gliner2 relations --relation works_for --relation located_in article.txt

# Structured JSON (structures file matches extract_json object shape)
gliner2 json --structures product_fields.json --text-split full product_blurb.txt

# Multitask: JSONL file, custom text field
gliner2 run --schema-file schema.json --text-field body --batch-size 4 docs.jsonl
```

Minimal multitask schema file (trimmed from fixtures):

```json
{
  "json_structures": [],
  "entities": { "company": "", "product": "" },
  "relations": [],
  "classifications": [
    {
      "task": "sentiment",
      "labels": ["positive", "negative", "neutral"],
      "multi_label": false,
      "cls_threshold": 0.5,
      "true_label": ["N/A"]
    }
  ]
}
```

## Python Interface (Not implemented yet)

A Python package that wraps this Rust implementation (`gliner2_rs`) is planned *if* we can get rust performance to be better than Python; it is **not implemented yet** (this section is a placeholder).

```bash
# use your package manager of choice
uv add gliner2_rs
```

```python
from gliner2_rs import Gliner2

gliner2 = Gliner2.from_pretrained('fastino/gliner2-base-v1')

text = "Apple CEO Tim Cook announced iPhone 15 in Cupertino yesterday."
result = extractor.extract_entities(text, ["company", "person", "product", "location"])

print(result)
# {'entities': {'company': ['Apple'], 'person': ['Tim Cook'], 'product': ['iPhone 15'], 'location': ['Cupertino']}}
```

