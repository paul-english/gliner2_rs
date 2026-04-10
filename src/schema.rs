//! Python-aligned `Schema` builder and extraction metadata (`gliner2.inference.engine.Schema`).

use anyhow::{Context, Result as AnyResult};
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::collections::HashMap;

/// Match mode for [`RegexValidator`] (Python `RegexValidator.mode`: `"full"` | `"partial"`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RegexMatchMode {
    /// `re.fullmatch` — entire span text must match.
    #[default]
    Full,
    /// `re.search` — pattern may match a substring.
    Partial,
}

/// Regex-based span filter for post-processing (Python `RegexValidator`).
#[derive(Clone, Debug)]
pub struct RegexValidator {
    re: Regex,
    mode: RegexMatchMode,
    exclude: bool,
}

impl RegexValidator {
    /// Build a validator. `case_insensitive` matches Python default `re.IGNORECASE`.
    pub fn new(
        pattern: impl AsRef<str>,
        mode: RegexMatchMode,
        exclude: bool,
        case_insensitive: bool,
    ) -> AnyResult<Self> {
        let re = regex::RegexBuilder::new(pattern.as_ref())
            .case_insensitive(case_insensitive)
            .build()
            .with_context(|| format!("Invalid regex: {:?}", pattern.as_ref()))?;
        Ok(Self { re, mode, exclude })
    }

    /// Python defaults: `mode=full`, `exclude=false`, `flags=re.IGNORECASE`.
    pub fn with_defaults(pattern: impl AsRef<str>) -> AnyResult<Self> {
        Self::new(pattern, RegexMatchMode::Full, false, true)
    }

    /// Returns whether the span text passes this validator (Python `validate`).
    pub fn validate(&self, text: &str) -> bool {
        let matched = match self.mode {
            RegexMatchMode::Full => self
                .re
                .find(text)
                .is_some_and(|m| m.start() == 0 && m.end() == text.len()),
            RegexMatchMode::Partial => self.re.is_match(text),
        };
        if self.exclude { !matched } else { matched }
    }
}

/// Field typing for structure / entity metadata (mirrors Python `dtype`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ValueDtype {
    Str,
    #[default]
    List,
}

/// Parsed `extract_json` field spec (string `name::dtype::[choices]::description` or JSON object).
#[derive(Clone, Debug)]
pub struct ParsedFieldSpec {
    pub name: String,
    pub dtype: ValueDtype,
    pub choices: Option<Vec<String>>,
    pub description: Option<String>,
    pub validators: Vec<RegexValidator>,
}

fn parse_validator_specs(arr: &[Value]) -> Vec<RegexValidator> {
    let mut out = Vec::new();
    for v in arr {
        let Some(o) = v.as_object() else {
            continue;
        };
        let Some(pattern) = o.get("pattern").and_then(|x| x.as_str()) else {
            continue;
        };
        if pattern.is_empty() {
            continue;
        }
        let mode = o
            .get("mode")
            .and_then(|x| x.as_str())
            .map(|s| match s {
                "partial" => RegexMatchMode::Partial,
                _ => RegexMatchMode::Full,
            })
            .unwrap_or(RegexMatchMode::Full);
        let exclude = o.get("exclude").and_then(|x| x.as_bool()).unwrap_or(false);
        let case_insensitive = o
            .get("case_insensitive")
            .and_then(|x| x.as_bool())
            .unwrap_or(true);
        if let Ok(val) = RegexValidator::new(pattern, mode, exclude, case_insensitive) {
            out.push(val);
        }
    }
    out
}

/// Parse a field spec string or object (Python `GLiNER2._parse_field_spec`).
pub fn parse_field_spec(spec: &Value) -> AnyResult<ParsedFieldSpec> {
    if let Some(o) = spec.as_object() {
        let name = o
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let dtype = o
            .get("dtype")
            .and_then(|v| v.as_str())
            .map(parse_field_dtype_token)
            .unwrap_or(ValueDtype::List);
        let choices = o.get("choices").and_then(|v| {
            v.as_array().map(|a| {
                a.iter()
                    .filter_map(|x| x.as_str().map(String::from))
                    .collect::<Vec<_>>()
            })
        });
        let description = o
            .get("description")
            .and_then(|v| v.as_str())
            .map(String::from);
        let validators = o
            .get("validators")
            .and_then(|v| v.as_array())
            .map(|a| parse_validator_specs(a))
            .unwrap_or_default();
        return Ok(ParsedFieldSpec {
            name,
            dtype,
            choices,
            description,
            validators,
        });
    }

    let s = spec
        .as_str()
        .context("field spec must be a string or a JSON object")?;
    parse_field_spec_str(s)
}

fn parse_field_dtype_token(s: &str) -> ValueDtype {
    match s {
        "str" => ValueDtype::Str,
        _ => ValueDtype::List,
    }
}

fn parse_field_spec_str(spec: &str) -> AnyResult<ParsedFieldSpec> {
    let parts: Vec<&str> = spec.split("::").collect();
    let name = parts.first().copied().unwrap_or("").trim().to_string();
    let mut dtype = ValueDtype::List;
    let mut choices: Option<Vec<String>> = None;
    let mut desc: Option<String> = None;
    let mut dtype_explicitly_set = false;

    if parts.len() <= 1 {
        return Ok(ParsedFieldSpec {
            name,
            dtype,
            choices,
            description: desc,
            validators: Vec::new(),
        });
    }

    for part in parts.iter().skip(1) {
        let part = *part;
        if part == "str" || part == "list" {
            dtype = parse_field_dtype_token(part);
            dtype_explicitly_set = true;
        } else if part.starts_with('[') && part.ends_with(']') && part.len() >= 2 {
            let inner = &part[1..part.len() - 1];
            choices = Some(
                inner
                    .split('|')
                    .map(|c| c.trim().to_string())
                    .filter(|c| !c.is_empty())
                    .collect(),
            );
            if !dtype_explicitly_set {
                dtype = ValueDtype::Str;
            }
        } else {
            desc = Some(part.to_string());
        }
    }

    Ok(ParsedFieldSpec {
        name,
        dtype,
        choices,
        description: desc,
        validators: Vec::new(),
    })
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct FieldMeta {
    pub dtype: ValueDtype,
    pub threshold: Option<f32>,
    pub choices: Option<Vec<String>>,
    /// Post-extraction span filters (not serialized in JSON schema dict; used in [`ExtractionMetadata`] only).
    #[serde(skip)]
    pub validators: Vec<RegexValidator>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct EntityMeta {
    pub dtype: ValueDtype,
    pub threshold: Option<f32>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RelationMeta {
    pub threshold: Option<f32>,
}

/// Metadata passed to decoders / `format_results` (mirrors `batch_extract` `metadata_list` entries).
#[derive(Clone, Debug, Default)]
pub struct ExtractionMetadata {
    pub field_metadata: HashMap<String, FieldMeta>,
    pub entity_metadata: HashMap<String, EntityMeta>,
    pub relation_metadata: HashMap<String, RelationMeta>,
    pub field_orders: HashMap<String, Vec<String>>,
    pub entity_order: Vec<String>,
    pub relation_order: Vec<String>,
    pub classification_tasks: Vec<String>,
}

/// Mutable schema builder; call `.build()` for a JSON value suitable for preprocessing.
#[derive(Debug)]
pub struct Schema {
    inner: Map<String, Value>,
    field_metadata: HashMap<String, FieldMeta>,
    entity_metadata: HashMap<String, EntityMeta>,
    relation_metadata: HashMap<String, RelationMeta>,
    field_orders: HashMap<String, Vec<String>>,
    entity_order: Vec<String>,
    relation_order: Vec<String>,
    active_structure: Option<StructureBuilderState>,
}

#[derive(Debug)]
struct StructureBuilderState {
    parent: String,
    fields: Map<String, Value>,
    field_order: Vec<String>,
    descriptions: Map<String, Value>,
}

impl Schema {
    pub fn new() -> Self {
        let mut inner = Map::new();
        inner.insert("json_structures".into(), json!([]));
        inner.insert("classifications".into(), json!([]));
        inner.insert("entities".into(), json!({}));
        inner.insert("relations".into(), json!([]));
        inner.insert("json_descriptions".into(), json!({}));
        inner.insert("entity_descriptions".into(), json!({}));
        Self {
            inner,
            field_metadata: HashMap::new(),
            entity_metadata: HashMap::new(),
            relation_metadata: HashMap::new(),
            field_orders: HashMap::new(),
            entity_order: Vec::new(),
            relation_order: Vec::new(),
            active_structure: None,
        }
    }

    fn finish_structure_if_any(&mut self) {
        if let Some(st) = self.active_structure.take() {
            self.field_orders.insert(st.parent.clone(), st.field_order);
            let arr = self
                .inner
                .get_mut("json_structures")
                .and_then(|v| v.as_array_mut())
                .expect("json_structures");
            let mut obj = Map::new();
            let parent = st.parent.clone();
            obj.insert(parent.clone(), Value::Object(st.fields));
            arr.push(Value::Object(obj));
            if !st.descriptions.is_empty() {
                let jd = self
                    .inner
                    .get_mut("json_descriptions")
                    .and_then(|v| v.as_object_mut())
                    .expect("json_descriptions");
                jd.insert(parent, Value::Object(st.descriptions));
            }
        }
    }

    pub fn structure(&mut self, name: impl Into<String>) -> StructureBuilder<'_> {
        self.finish_structure_if_any();
        let parent = name.into();
        self.active_structure = Some(StructureBuilderState {
            parent: parent.clone(),
            fields: Map::new(),
            field_order: Vec::new(),
            descriptions: Map::new(),
        });
        StructureBuilder { schema: self }
    }

    /// Finish an open [`StructureBuilder`] without starting another task (for explicit control).
    pub fn finish_structure(&mut self) {
        self.finish_structure_if_any();
    }

    /// Add structures from `extract_json`-style maps: `{ "parent": ["field::str::...", ...], ... }`.
    pub fn extract_json_structures(&mut self, structures: &Map<String, Value>) -> AnyResult<()> {
        for (parent, fields_val) in structures {
            let arr = fields_val.as_array().with_context(|| {
                format!("extract_json_structures: {parent:?} must be a JSON array")
            })?;
            let mut builder = self.structure(parent.clone());
            for spec in arr {
                let p = parse_field_spec(spec).with_context(|| {
                    format!("extract_json_structures: invalid field spec under {parent:?}")
                })?;
                builder.field(
                    p.name,
                    p.dtype,
                    p.choices,
                    p.description,
                    None,
                    Some(p.validators).filter(|v| !v.is_empty()),
                );
            }
        }
        Ok(())
    }

    /// `entity_types`: names, or map name -> description string / config object.
    pub fn entities(&mut self, entity_types: Value) -> &mut Self {
        self.finish_structure_if_any();
        match entity_types {
            Value::String(s) => {
                let entities = self
                    .inner
                    .get_mut("entities")
                    .and_then(|v| v.as_object_mut())
                    .expect("entities");
                if !self.entity_order.contains(&s) {
                    self.entity_order.push(s.clone());
                }
                entities.insert(s.clone(), json!(""));
                self.entity_metadata.insert(
                    s,
                    EntityMeta {
                        dtype: ValueDtype::List,
                        threshold: None,
                    },
                );
            }
            Value::Array(arr) => {
                let entities = self
                    .inner
                    .get_mut("entities")
                    .and_then(|v| v.as_object_mut())
                    .expect("entities");
                for v in arr {
                    let name = v.as_str().unwrap_or("").to_string();
                    if name.is_empty() {
                        continue;
                    }
                    if !self.entity_order.contains(&name) {
                        self.entity_order.push(name.clone());
                    }
                    entities.insert(name.clone(), json!(""));
                    self.entity_metadata.insert(
                        name,
                        EntityMeta {
                            dtype: ValueDtype::List,
                            threshold: None,
                        },
                    );
                }
            }
            Value::Object(map) => {
                let mut ent_map = self
                    .inner
                    .remove("entities")
                    .and_then(|v| v.as_object().cloned())
                    .unwrap_or_default();
                let mut desc_map = self
                    .inner
                    .remove("entity_descriptions")
                    .and_then(|v| v.as_object().cloned())
                    .unwrap_or_default();
                for (name, cfg) in map {
                    if !self.entity_order.contains(&name) {
                        self.entity_order.push(name.clone());
                    }
                    let (description, dtype, threshold) = parse_entity_config(&cfg);
                    if let Some(d) = description {
                        desc_map.insert(name.clone(), json!(d));
                    }
                    ent_map.insert(name.clone(), json!(""));
                    self.entity_metadata.insert(
                        name.clone(),
                        EntityMeta {
                            dtype: dtype.unwrap_or(ValueDtype::List),
                            threshold,
                        },
                    );
                }
                self.inner.insert("entities".into(), Value::Object(ent_map));
                self.inner
                    .insert("entity_descriptions".into(), Value::Object(desc_map));
            }
            _ => {}
        }
        self
    }

    /// Single-label classification with default `cls_threshold` 0.5.
    pub fn classification_simple(&mut self, task: impl Into<String>, labels: Value) -> &mut Self {
        self.classification(task, labels, false, 0.5)
    }

    pub fn classification(
        &mut self,
        task: impl Into<String>,
        labels: Value,
        multi_label: bool,
        cls_threshold: f32,
    ) -> &mut Self {
        self.finish_structure_if_any();
        let task = task.into();
        let (label_names, label_descs) = parse_labels(labels);
        let mut cfg = Map::new();
        cfg.insert("task".into(), json!(task.clone()));
        cfg.insert("labels".into(), json!(label_names.clone()));
        cfg.insert("multi_label".into(), json!(multi_label));
        cfg.insert(
            "cls_threshold".into(),
            serde_json::Number::from_f64(cls_threshold as f64)
                .unwrap()
                .into(),
        );
        cfg.insert("true_label".into(), json!(["N/A"]));
        if let Some(descs) = label_descs {
            cfg.insert("label_descriptions".into(), Value::Object(descs));
        }
        self.inner
            .get_mut("classifications")
            .and_then(|v| v.as_array_mut())
            .expect("classifications")
            .push(Value::Object(cfg));
        self
    }

    pub fn relations(&mut self, relation_types: Value) -> &mut Self {
        self.finish_structure_if_any();
        let rels = self
            .inner
            .get_mut("relations")
            .and_then(|v| v.as_array_mut())
            .expect("relations");

        let mut add_one = |name: String, threshold: Option<f32>| {
            let mut inner = Map::new();
            let mut fields = Map::new();
            fields.insert("head".into(), json!(""));
            fields.insert("tail".into(), json!(""));
            inner.insert(name.clone(), Value::Object(fields));
            rels.push(Value::Object(inner));
            if !self.relation_order.contains(&name) {
                self.relation_order.push(name.clone());
            }
            self.field_orders
                .insert(name.clone(), vec!["head".into(), "tail".into()]);
            self.relation_metadata
                .insert(name, RelationMeta { threshold });
        };

        match relation_types {
            Value::String(s) => add_one(s, None),
            Value::Array(arr) => {
                for v in arr {
                    if let Some(s) = v.as_str() {
                        add_one(s.to_string(), None);
                    }
                }
            }
            Value::Object(map) => {
                for (name, cfg) in map {
                    let (desc, threshold) = parse_relation_config(&cfg);
                    if desc.is_some() {
                        // Python stores description in relation metadata only indirectly;
                        // keep relation row as empty head/tail.
                    }
                    add_one(name, threshold);
                }
            }
            _ => {}
        }
        self
    }

    pub fn build(&mut self) -> (Value, ExtractionMetadata) {
        self.finish_structure_if_any();
        let classification_tasks: Vec<String> = self
            .inner
            .get("classifications")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|c| c.get("task").and_then(|t| t.as_str()).map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        let meta = ExtractionMetadata {
            field_metadata: self.field_metadata.clone(),
            entity_metadata: self.entity_metadata.clone(),
            relation_metadata: self.relation_metadata.clone(),
            field_orders: self.field_orders.clone(),
            entity_order: self.entity_order.clone(),
            relation_order: self.relation_order.clone(),
            classification_tasks,
        };
        (Value::Object(self.inner.clone()), meta)
    }
}

pub struct StructureBuilder<'a> {
    schema: &'a mut Schema,
}

impl Drop for StructureBuilder<'_> {
    fn drop(&mut self) {
        self.schema.finish_structure_if_any();
    }
}

impl<'a> StructureBuilder<'a> {
    pub fn field_str(&mut self, name: impl Into<String>) -> &mut Self {
        self.field(name, ValueDtype::Str, None, None, None, None)
    }

    pub fn field_list(&mut self, name: impl Into<String>) -> &mut Self {
        self.field(name, ValueDtype::List, None, None, None, None)
    }

    pub fn field_choices(
        &mut self,
        name: impl Into<String>,
        choices: Vec<String>,
        dtype: ValueDtype,
    ) -> &mut Self {
        self.field(name, dtype, Some(choices), None, None, None)
    }

    pub fn field(
        &mut self,
        name: impl Into<String>,
        dtype: ValueDtype,
        choices: Option<Vec<String>>,
        description: Option<String>,
        threshold: Option<f32>,
        validators: Option<Vec<RegexValidator>>,
    ) -> &mut Self {
        let name = name.into();
        let st = self.schema.active_structure.as_mut().expect("structure");
        st.field_order.push(name.clone());
        let v = match (&choices, dtype) {
            (Some(c), ValueDtype::Str) => json!({ "value": "", "choices": c, "dtype": "str" }),
            (Some(c), ValueDtype::List) => json!({ "value": "", "choices": c }),
            (None, ValueDtype::Str) => json!({ "dtype": "str" }),
            (None, ValueDtype::List) => json!(""),
        };
        st.fields.insert(name.clone(), v);
        if let Some(d) = description {
            st.descriptions.insert(name.clone(), json!(d));
        }
        self.schema.field_metadata.insert(
            format!("{}.{}", st.parent, name),
            FieldMeta {
                dtype,
                threshold,
                choices,
                validators: validators.unwrap_or_default(),
            },
        );
        self
    }
}

impl Default for Schema {
    fn default() -> Self {
        Self::new()
    }
}

fn parse_entity_config(cfg: &Value) -> (Option<String>, Option<ValueDtype>, Option<f32>) {
    match cfg {
        Value::String(d) => (Some(d.clone()), None, None),
        Value::Object(m) => {
            let desc = m
                .get("description")
                .and_then(|v| v.as_str())
                .map(String::from);
            let dtype = m
                .get("dtype")
                .and_then(|v| v.as_str())
                .and_then(|s| match s {
                    "str" => Some(ValueDtype::Str),
                    "list" => Some(ValueDtype::List),
                    _ => None,
                });
            let threshold = m
                .get("threshold")
                .and_then(|v| v.as_f64())
                .map(|f| f as f32);
            (desc, dtype, threshold)
        }
        _ => (None, None, None),
    }
}

fn parse_relation_config(cfg: &Value) -> (Option<String>, Option<f32>) {
    match cfg {
        Value::String(d) => (Some(d.clone()), None),
        Value::Object(m) => {
            let desc = m
                .get("description")
                .and_then(|v| v.as_str())
                .map(String::from);
            let threshold = m
                .get("threshold")
                .and_then(|v| v.as_f64())
                .map(|f| f as f32);
            (desc, threshold)
        }
        _ => (None, None),
    }
}

fn parse_labels(labels: Value) -> (Vec<String>, Option<Map<String, Value>>) {
    match labels {
        Value::Array(arr) => (
            arr.into_iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect(),
            None,
        ),
        Value::Object(map) => {
            let names: Vec<String> = map.keys().cloned().collect();
            (names, Some(map))
        }
        _ => (Vec::new(), None),
    }
}

/// Start a new schema (equivalent to `GLiNER2.create_schema()`).
pub fn create_schema() -> Schema {
    Schema::new()
}

/// Metadata for a plain schema dict (mirrors Python `batch_extract` when `schema` is a `dict`, not a `Schema` instance).
pub fn infer_metadata_from_schema(schema: &Value) -> ExtractionMetadata {
    let mut meta = ExtractionMetadata::default();

    if let Some(ent) = schema.get("entities").and_then(|v| v.as_object()) {
        meta.entity_order = ent.keys().cloned().collect();
    }

    if let Some(arr) = schema.get("classifications").and_then(|v| v.as_array()) {
        meta.classification_tasks = arr
            .iter()
            .filter_map(|c| c.get("task").and_then(|t| t.as_str()).map(String::from))
            .collect();
    }

    if let Some(arr) = schema.get("json_structures").and_then(|v| v.as_array()) {
        for st in arr {
            if let Some(obj) = st.as_object() {
                for (parent, fields) in obj {
                    if let Some(fmap) = fields.as_object() {
                        meta.field_orders
                            .insert(parent.clone(), fmap.keys().cloned().collect());
                        for (fname, fval) in fmap {
                            let key = format!("{parent}.{fname}");
                            let dtype = fval
                                .get("dtype")
                                .and_then(|v| v.as_str())
                                .map(|s| match s {
                                    "str" => ValueDtype::Str,
                                    _ => ValueDtype::List,
                                })
                                .unwrap_or(ValueDtype::List);
                            let threshold = fval
                                .get("threshold")
                                .and_then(|v| v.as_f64())
                                .map(|f| f as f32);
                            let choices = fval.get("choices").and_then(|c| c.as_array()).map(|a| {
                                a.iter()
                                    .filter_map(|v| v.as_str().map(String::from))
                                    .collect::<Vec<_>>()
                            });
                            let validators = fval
                                .as_object()
                                .and_then(|o| o.get("validators"))
                                .and_then(|v| v.as_array())
                                .map(|a| parse_validator_specs(a))
                                .unwrap_or_default();
                            meta.field_metadata.insert(
                                key.clone(),
                                FieldMeta {
                                    dtype,
                                    threshold,
                                    choices,
                                    validators,
                                },
                            );
                        }
                    }
                }
            }
        }
    }

    if let Some(arr) = schema.get("relations").and_then(|v| v.as_array()) {
        for item in arr {
            if let Some(obj) = item.as_object() {
                for (rel_name, _) in obj {
                    if !meta.relation_order.contains(rel_name) {
                        meta.relation_order.push(rel_name.clone());
                    }
                    meta.field_orders
                        .insert(rel_name.clone(), vec!["head".into(), "tail".into()]);
                }
            }
        }
    }

    meta
}

// ---------------------------------------------------------------------------
// Schema introspection
// ---------------------------------------------------------------------------

/// Description of a single entity type in a schema.
#[derive(Clone, Debug)]
pub struct EntityTypeInfo {
    pub name: String,
    pub description: Option<String>,
}

/// Description of a single relation type.
#[derive(Clone, Debug)]
pub struct RelationTypeInfo {
    pub name: String,
}

/// Description of a classification task.
#[derive(Clone, Debug)]
pub struct ClassificationTaskInfo {
    pub task: String,
    pub labels: Vec<String>,
    pub multi_label: bool,
}

/// Description of a field within a JSON structure.
#[derive(Clone, Debug)]
pub struct StructureFieldInfo {
    pub name: String,
    pub dtype: ValueDtype,
    pub choices: Option<Vec<String>>,
}

/// Description of a JSON structure.
#[derive(Clone, Debug)]
pub struct StructureInfo {
    pub name: String,
    pub fields: Vec<StructureFieldInfo>,
}

/// All task types present in a schema, extracted for consumers that need to
/// enumerate what a schema defines (e.g. generating Label Studio configs).
#[derive(Clone, Debug, Default)]
pub struct SchemaInfo {
    pub entities: Vec<EntityTypeInfo>,
    pub relations: Vec<RelationTypeInfo>,
    pub classifications: Vec<ClassificationTaskInfo>,
    pub structures: Vec<StructureInfo>,
}

impl SchemaInfo {
    /// Extract structured task information from a built schema JSON value.
    pub fn from_value(schema: &Value) -> Self {
        let mut info = SchemaInfo::default();

        // Entities
        if let Some(ent_obj) = schema.get("entities").and_then(|v| v.as_object()) {
            let desc_obj = schema
                .get("entity_descriptions")
                .and_then(|v| v.as_object());
            for name in ent_obj.keys() {
                let description = desc_obj
                    .and_then(|d| d.get(name))
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                    .map(String::from);
                info.entities.push(EntityTypeInfo {
                    name: name.clone(),
                    description,
                });
            }
        }

        // Relations
        if let Some(arr) = schema.get("relations").and_then(|v| v.as_array()) {
            for item in arr {
                if let Some(obj) = item.as_object() {
                    for rel_name in obj.keys() {
                        info.relations.push(RelationTypeInfo {
                            name: rel_name.clone(),
                        });
                    }
                }
            }
        }

        // Classifications
        if let Some(arr) = schema.get("classifications").and_then(|v| v.as_array()) {
            for cls in arr {
                let task = cls
                    .get("task")
                    .and_then(|t| t.as_str())
                    .unwrap_or("")
                    .to_string();
                let labels = cls
                    .get("labels")
                    .and_then(|v| v.as_array())
                    .map(|a| {
                        a.iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default();
                let multi_label = cls
                    .get("multi_label")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                if !task.is_empty() {
                    info.classifications.push(ClassificationTaskInfo {
                        task,
                        labels,
                        multi_label,
                    });
                }
            }
        }

        // JSON structures
        if let Some(arr) = schema.get("json_structures").and_then(|v| v.as_array()) {
            for st in arr {
                if let Some(obj) = st.as_object() {
                    for (parent, fields_val) in obj {
                        let mut fields = Vec::new();
                        if let Some(fmap) = fields_val.as_object() {
                            for (fname, fval) in fmap {
                                let (dtype, choices) = match fval {
                                    Value::String(_) => (ValueDtype::List, None),
                                    Value::Object(o) => {
                                        let dt = o
                                            .get("dtype")
                                            .and_then(|v| v.as_str())
                                            .map(|s| match s {
                                                "str" => ValueDtype::Str,
                                                _ => ValueDtype::List,
                                            })
                                            .unwrap_or(ValueDtype::List);
                                        let ch = o
                                            .get("choices")
                                            .and_then(|v| v.as_array())
                                            .map(|a| {
                                                a.iter()
                                                    .filter_map(|v| v.as_str().map(String::from))
                                                    .collect()
                                            });
                                        (dt, ch)
                                    }
                                    _ => (ValueDtype::List, None),
                                };
                                fields.push(StructureFieldInfo {
                                    name: fname.clone(),
                                    dtype,
                                    choices,
                                });
                            }
                        }
                        info.structures.push(StructureInfo {
                            name: parent.clone(),
                            fields,
                        });
                    }
                }
            }
        }

        info
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn infer_metadata_entity_order() {
        let v = json!({
            "entities": { "a": "", "b": "" },
            "classifications": [],
            "json_structures": [],
            "relations": []
        });
        let m = infer_metadata_from_schema(&v);
        assert_eq!(m.entity_order, vec!["a", "b"]);
    }

    #[test]
    fn parse_field_restaurant() {
        let p = parse_field_spec(&json!("restaurant::str::Restaurant name")).unwrap();
        assert_eq!(p.name, "restaurant");
        assert_eq!(p.dtype, ValueDtype::Str);
        assert!(p.choices.is_none());
        assert_eq!(p.description.as_deref(), Some("Restaurant name"));
    }

    #[test]
    fn parse_field_seating() {
        let p =
            parse_field_spec(&json!("seating::[indoor|outdoor|bar]::Seating preference")).unwrap();
        assert_eq!(p.name, "seating");
        assert_eq!(p.dtype, ValueDtype::Str);
        assert_eq!(
            p.choices,
            Some(vec!["indoor".into(), "outdoor".into(), "bar".into()])
        );
        assert_eq!(p.description.as_deref(), Some("Seating preference"));
    }

    #[test]
    fn parse_field_dietary_list() {
        let p = parse_field_spec(&json!(
            "dietary::[vegetarian|vegan|gluten-free|none]::list::Dietary restrictions"
        ))
        .unwrap();
        assert_eq!(p.name, "dietary");
        assert_eq!(p.dtype, ValueDtype::List);
        assert_eq!(p.choices.as_ref().unwrap().len(), 4);
        assert_eq!(p.description.as_deref(), Some("Dietary restrictions"));
    }

    #[test]
    fn parse_field_dict() {
        let p = parse_field_spec(&json!({
            "name": "x",
            "dtype": "str",
            "choices": ["a", "b"],
            "description": "d"
        }))
        .unwrap();
        assert_eq!(p.name, "x");
        assert_eq!(p.dtype, ValueDtype::Str);
        assert_eq!(p.choices, Some(vec!["a".into(), "b".into()]));
        assert_eq!(p.description.as_deref(), Some("d"));
    }

    #[test]
    fn extract_json_structures_builds_schema() {
        let mut s = Schema::new();
        let m = serde_json::json!({
            "product_info": [
                "name::str",
                "price::str",
                "features::list",
                "availability::str::[in_stock|pre_order|sold_out]"
            ]
        })
        .as_object()
        .unwrap()
        .clone();
        s.extract_json_structures(&m).unwrap();
        let (v, meta) = s.build();
        let arr = v["json_structures"].as_array().unwrap();
        assert_eq!(arr.len(), 1);
        let fields = arr[0]["product_info"].as_object().unwrap();
        assert!(fields.contains_key("name"));
        assert!(fields["availability"].get("choices").is_some());
        assert!(meta.field_orders.contains_key("product_info"));
    }

    #[test]
    fn regex_validator_full_partial_exclude() {
        let v = RegexValidator::new(r"^\d{3}$", RegexMatchMode::Full, false, false).unwrap();
        assert!(v.validate("123"));
        assert!(!v.validate("1234"));
        assert!(!v.validate("a12"));

        // Partial uses `search` — unanchored pattern finds digits inside a longer string.
        let partial = RegexValidator::new(r"\d{3}", RegexMatchMode::Partial, false, false).unwrap();
        assert!(partial.validate("x123y"));

        let ex = RegexValidator::new(r"^test", RegexMatchMode::Partial, true, false).unwrap();
        assert!(!ex.validate("tester"));
        assert!(ex.validate("hello"));
    }

    #[test]
    fn regex_validator_default_case_insensitive() {
        let v = RegexValidator::with_defaults(r"^[A-Z]+$").unwrap();
        assert!(v.validate("abc"));
    }

    #[test]
    fn structure_builder_field_validators_in_metadata() {
        let email = RegexValidator::with_defaults(r"^[\w.-]+@[\w.-]+\.\w+$").unwrap();
        let mut s = Schema::new();
        {
            let mut b = s.structure("contact");
            b.field(
                "email",
                ValueDtype::Str,
                None,
                None,
                None,
                Some(vec![email]),
            );
        }
        let (_v, meta) = s.build();
        let fm = meta
            .field_metadata
            .get("contact.email")
            .expect("contact.email");
        assert_eq!(fm.validators.len(), 1);
        assert!(fm.validators[0].validate("a@b.co"));
        assert!(!fm.validators[0].validate("not-an-email"));
    }

    #[test]
    fn infer_metadata_parses_validators() {
        let v = json!({
            "entities": {},
            "classifications": [],
            "json_structures": [{
                "form": {
                    "code": {
                        "dtype": "str",
                        "validators": [
                            { "pattern": r"^\d{3}$", "mode": "full", "exclude": false, "case_insensitive": true }
                        ]
                    }
                }
            }],
            "relations": []
        });
        let m = infer_metadata_from_schema(&v);
        let fm = m.field_metadata.get("form.code").expect("form.code");
        assert_eq!(fm.validators.len(), 1);
        assert!(fm.validators[0].validate("042"));
        assert!(!fm.validators[0].validate("42"));
    }

    #[test]
    fn parse_field_spec_with_validators() {
        let p = parse_field_spec(&json!({
            "name": "email",
            "dtype": "str",
            "validators": [
                { "pattern": "@", "mode": "partial" }
            ]
        }))
        .unwrap();
        assert_eq!(p.name, "email");
        assert_eq!(p.validators.len(), 1);
        assert!(p.validators[0].validate("a@b"));
    }

    #[test]
    fn introspect_entities_with_descriptions() {
        let mut s = Schema::new();
        s.entities(json!({"person": "A person's name", "org": "An organization"}));
        let (v, _) = s.build();
        let info = SchemaInfo::from_value(&v);
        assert_eq!(info.entities.len(), 2);
        let person = info.entities.iter().find(|e| e.name == "person").unwrap();
        assert_eq!(person.description.as_deref(), Some("A person's name"));
    }

    #[test]
    fn introspect_entities_simple() {
        let mut s = Schema::new();
        s.entities(json!(["person", "org"]));
        let (v, _) = s.build();
        let info = SchemaInfo::from_value(&v);
        assert_eq!(info.entities.len(), 2);
        assert!(info.entities[0].description.is_none());
    }

    #[test]
    fn introspect_relations() {
        let mut s = Schema::new();
        s.relations(json!(["works_for", "located_in"]));
        let (v, _) = s.build();
        let info = SchemaInfo::from_value(&v);
        assert_eq!(info.relations.len(), 2);
        assert_eq!(info.relations[0].name, "works_for");
        assert_eq!(info.relations[1].name, "located_in");
    }

    #[test]
    fn introspect_classifications() {
        let mut s = Schema::new();
        s.classification("sentiment", json!(["positive", "negative"]), false, 0.5);
        s.classification("topic", json!(["tech", "sports", "politics"]), true, 0.3);
        let (v, _) = s.build();
        let info = SchemaInfo::from_value(&v);
        assert_eq!(info.classifications.len(), 2);
        assert_eq!(info.classifications[0].task, "sentiment");
        assert_eq!(info.classifications[0].labels, vec!["positive", "negative"]);
        assert!(!info.classifications[0].multi_label);
        assert_eq!(info.classifications[1].task, "topic");
        assert!(info.classifications[1].multi_label);
    }

    #[test]
    fn introspect_structures() {
        let mut s = Schema::new();
        let m = json!({
            "product_info": [
                "name::str",
                "features::list",
                "availability::str::[in_stock|pre_order|sold_out]"
            ]
        })
        .as_object()
        .unwrap()
        .clone();
        s.extract_json_structures(&m).unwrap();
        let (v, _) = s.build();
        let info = SchemaInfo::from_value(&v);
        assert_eq!(info.structures.len(), 1);
        assert_eq!(info.structures[0].name, "product_info");
        assert_eq!(info.structures[0].fields.len(), 3);
        let avail = info.structures[0]
            .fields
            .iter()
            .find(|f| f.name == "availability")
            .unwrap();
        assert_eq!(avail.dtype, ValueDtype::Str);
        assert_eq!(
            avail.choices.as_deref().unwrap(),
            &["in_stock", "pre_order", "sold_out"]
        );
    }

    #[test]
    fn introspect_multitask() {
        let mut s = Schema::new();
        s.entities(json!(["person"]));
        s.relations(json!(["works_for"]));
        s.classification_simple("sentiment", json!(["pos", "neg"]));
        let (v, _) = s.build();
        let info = SchemaInfo::from_value(&v);
        assert_eq!(info.entities.len(), 1);
        assert_eq!(info.relations.len(), 1);
        assert_eq!(info.classifications.len(), 1);
        assert!(info.structures.is_empty());
    }
}
