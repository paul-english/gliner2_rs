# Gliner2 Rust

This project implements the Gliner2 model in rust with compatibility to the original weights and output of the python training.

```bash
cargo add gliner2
# and/or for a cli utility
cargo install gliner2
```

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

### Batch Inference (Not implemented yet)

We plan to support batch inference similar to the python implementation soon.

## CLI Usage (Not implemented yet)

The CLI allows running batch inference tasks against individual files (json,jsonl,txt).

```bash
# entity extraction
gliner2 extract --label PER --label ORG <input>
gliner2 extract --label "PER:A person" --label "ORG:An organization" --confidence <input>

# classify
gliner2 classify --multi --label ...

# schema extraction
gliner2 schema --schema "{...}"

# relation extraction
gliner2 relation --relation works_for
```

Inputs:
- `jsonl`: gliner2_rs will default to looking for a `text` field, though this can be overridden with the `--field` argument
- `json`: Similar to `jsonl` for a single record. (Use `jq` or similar if you need batching on a nested array in your json file).
- `txt`: Plain text input requires a segmentation method: `--segment-text=full|sentence|chars` (TODO needs some thought)

Output:
gliner2_rs will return `jsonl` output that matches the python output for any of these tasks.

## Python Interface (Not implemented yet)

This faster version of gliner2 is available in a python package as well for speeding up any existing python inference implementations.

``` bash
# use your package manager of choice
uv add gliner2_rs
```

``` python
from gliner2_rs import Gliner2

gliner2 = Gliner2.from_pretrained('fastino/gliner2-base-v1')

text = "Apple CEO Tim Cook announced iPhone 15 in Cupertino yesterday."
result = extractor.extract_entities(text, ["company", "person", "product", "location"])

print(result)
# {'entities': {'company': ['Apple'], 'person': ['Tim Cook'], 'product': ['iPhone 15'], 'location': ['Cupertino']}}
```
