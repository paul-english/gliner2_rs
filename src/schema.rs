//! Python-aligned `Schema` builder and extraction metadata (`gliner2.inference.engine.Schema`).

use anyhow::{Context, Result as AnyResult};
use indexmap::IndexMap;
use regex::Regex;
use serde::{Deserialize, Serialize};
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

mod schema_document;
pub use schema_document::*;

/// Parsed `extract_json` field spec (string `name::dtype::[choices]::description` or JSON object).
#[derive(Clone, Debug)]
pub struct ParsedFieldSpec {
    pub name: String,
    pub dtype: ValueDtype,
    pub choices: Option<Vec<String>>,
    pub description: Option<String>,
    pub validators: Vec<RegexValidator>,
}

fn parse_validator_specs(arr: &[ValidatorSpec]) -> Vec<RegexValidator> {
    let mut out = Vec::new();
    for spec in arr {
        if spec.pattern.is_empty() {
            continue;
        }
        let mode = spec
            .mode
            .as_deref()
            .map(|s| match s {
                "partial" => RegexMatchMode::Partial,
                _ => RegexMatchMode::Full,
            })
            .unwrap_or(RegexMatchMode::Full);
        let exclude = spec.exclude.unwrap_or(false);
        let case_insensitive = spec.case_insensitive.unwrap_or(true);
        if let Ok(val) = RegexValidator::new(&spec.pattern, mode, exclude, case_insensitive) {
            out.push(val);
        }
    }
    out
}

/// Parse a field spec string or object (Python `GLiNER2._parse_field_spec`).
pub fn parse_field_spec(spec: &FieldSpecSource) -> AnyResult<ParsedFieldSpec> {
    match spec {
        FieldSpecSource::Obj(o) => {
            let dtype = o
                .dtype
                .as_deref()
                .map(parse_field_dtype_token)
                .unwrap_or(ValueDtype::List);
            let validators = o
                .validators
                .as_ref()
                .map(|a| parse_validator_specs(a))
                .unwrap_or_default();
            Ok(ParsedFieldSpec {
                name: o.name.clone(),
                dtype,
                choices: o.choices.clone(),
                description: o.description.clone(),
                validators,
            })
        }
        FieldSpecSource::Str(s) => parse_field_spec_str(s),
    }
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

/// Mutable schema builder; call `.build()` for an [`ExtractionSchema`] suitable for preprocessing.
#[derive(Debug)]
pub struct Schema {
    document: ExtractionSchema,
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
    fields: IndexMap<String, StructureField>,
    field_order: Vec<String>,
    descriptions: IndexMap<String, String>,
}

impl Schema {
    pub fn new() -> Self {
        Self {
            document: ExtractionSchema::default(),
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
            let parent = st.parent.clone();
            let mut parents = IndexMap::new();
            parents.insert(parent.clone(), st.fields);
            self.document
                .json_structures
                .push(JsonStructureBlock { parents });
            if !st.descriptions.is_empty() {
                self.document
                    .json_descriptions
                    .insert(parent, st.descriptions);
            }
        }
    }

    pub fn structure(&mut self, name: impl Into<String>) -> StructureBuilder<'_> {
        self.finish_structure_if_any();
        let parent = name.into();
        self.active_structure = Some(StructureBuilderState {
            parent: parent.clone(),
            fields: IndexMap::new(),
            field_order: Vec::new(),
            descriptions: IndexMap::new(),
        });
        StructureBuilder { schema: self }
    }

    /// Finish an open [`StructureBuilder`] without starting another task (for explicit control).
    pub fn finish_structure(&mut self) {
        self.finish_structure_if_any();
    }

    /// Add structures from `extract_json`-style maps: `{ "parent": ["field::str::...", ...], ... }`.
    pub fn extract_json_structures(
        &mut self,
        structures: &IndexMap<String, Vec<FieldSpecSource>>,
    ) -> AnyResult<()> {
        for (parent, arr) in structures {
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

    /// `entity_types`: names, or map name → description string / config object.
    pub fn entities(&mut self, entity_types: EntityTypesInput) -> &mut Self {
        self.finish_structure_if_any();
        match entity_types {
            EntityTypesInput::One(s) => {
                if !self.entity_order.contains(&s) {
                    self.entity_order.push(s.clone());
                }
                self.document.entities.insert(s.clone(), String::new());
                self.entity_metadata.insert(
                    s,
                    EntityMeta {
                        dtype: ValueDtype::List,
                        threshold: None,
                    },
                );
            }
            EntityTypesInput::Many(arr) => {
                for name in arr {
                    if name.is_empty() {
                        continue;
                    }
                    if !self.entity_order.contains(&name) {
                        self.entity_order.push(name.clone());
                    }
                    self.document.entities.insert(name.clone(), String::new());
                    self.entity_metadata.insert(
                        name,
                        EntityMeta {
                            dtype: ValueDtype::List,
                            threshold: None,
                        },
                    );
                }
            }
            EntityTypesInput::WithMeta(map) => {
                for (name, cfg) in map {
                    if !self.entity_order.contains(&name) {
                        self.entity_order.push(name.clone());
                    }
                    let (description, dtype, threshold) = match cfg {
                        EntityTypeConfigInput::DescriptionOnly(d) => (Some(d), None, None),
                        EntityTypeConfigInput::Full(c) => (c.description, c.dtype, c.threshold),
                    };
                    if let Some(d) = description {
                        self.document.entity_descriptions.insert(name.clone(), d);
                    }
                    self.document.entities.insert(name.clone(), String::new());
                    self.entity_metadata.insert(
                        name.clone(),
                        EntityMeta {
                            dtype: dtype.unwrap_or(ValueDtype::List),
                            threshold,
                        },
                    );
                }
            }
        }
        self
    }

    /// Single-label classification with default `cls_threshold` 0.5.
    pub fn classification_simple(
        &mut self,
        task: impl Into<String>,
        labels: ClassificationLabelsInput,
    ) -> &mut Self {
        self.classification(task, labels, false, 0.5)
    }

    pub fn classification(
        &mut self,
        task: impl Into<String>,
        labels: ClassificationLabelsInput,
        multi_label: bool,
        cls_threshold: f32,
    ) -> &mut Self {
        self.finish_structure_if_any();
        let task = task.into();
        let (label_names, label_descs) = match labels {
            ClassificationLabelsInput::List(v) => (v, None),
            ClassificationLabelsInput::WithDescriptions(m) => {
                let names: Vec<String> = m.keys().cloned().collect();
                (names, Some(m))
            }
        };
        self.document.classifications.push(ClassificationConfig {
            task,
            labels: label_names,
            multi_label,
            cls_threshold,
            true_label: Some(vec!["N/A".into()]),
            label_descriptions: label_descs,
            examples: None,
            class_act: None,
        });
        self
    }

    pub fn relations(&mut self, relation_types: RelationTypesInput) -> &mut Self {
        self.finish_structure_if_any();

        let add_one = |this: &mut Self, name: String, threshold: Option<f32>| {
            let mut types = IndexMap::new();
            types.insert(
                name.clone(),
                RelationEndpoints {
                    head: String::new(),
                    tail: String::new(),
                },
            );
            this.document.relations.push(RelationBlock { types });
            if !this.relation_order.contains(&name) {
                this.relation_order.push(name.clone());
            }
            this.field_orders
                .insert(name.clone(), vec!["head".into(), "tail".into()]);
            this.relation_metadata
                .insert(name, RelationMeta { threshold });
        };

        match relation_types {
            RelationTypesInput::One(s) => add_one(self, s, None),
            RelationTypesInput::Many(arr) => {
                for s in arr {
                    add_one(self, s, None);
                }
            }
            RelationTypesInput::WithMeta(map) => {
                for (name, cfg) in map {
                    let (_desc, threshold) = match cfg {
                        RelationTypeConfigInput::DescriptionOnly(_d) => (None, None),
                        RelationTypeConfigInput::Full(c) => (c.description, c.threshold),
                    };
                    add_one(self, name, threshold);
                }
            }
        }
        self
    }

    pub fn build(&mut self) -> (ExtractionSchema, ExtractionMetadata) {
        self.finish_structure_if_any();
        let classification_tasks = self
            .document
            .classifications
            .iter()
            .map(|c| c.task.clone())
            .collect();

        let meta = ExtractionMetadata {
            field_metadata: self.field_metadata.clone(),
            entity_metadata: self.entity_metadata.clone(),
            relation_metadata: self.relation_metadata.clone(),
            field_orders: self.field_orders.clone(),
            entity_order: self.entity_order.clone(),
            relation_order: self.relation_order.clone(),
            classification_tasks,
        };
        (self.document.clone(), meta)
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
            (Some(c), ValueDtype::Str) => StructureField::Rich(StructureFieldBody {
                value: Some(FieldDefault::Str(String::new())),
                choices: Some(c.clone()),
                dtype: Some(ValueDtype::Str),
                threshold: None,
                validators: vec![],
            }),
            (Some(c), ValueDtype::List) => StructureField::Rich(StructureFieldBody {
                value: Some(FieldDefault::Str(String::new())),
                choices: Some(c.clone()),
                dtype: None,
                threshold: None,
                validators: vec![],
            }),
            (None, ValueDtype::Str) => StructureField::Rich(StructureFieldBody {
                value: None,
                choices: None,
                dtype: Some(ValueDtype::Str),
                threshold: None,
                validators: vec![],
            }),
            (None, ValueDtype::List) => StructureField::Plain(String::new()),
        };
        st.fields.insert(name.clone(), v);
        if let Some(d) = description {
            st.descriptions.insert(name.clone(), d);
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

/// Start a new schema (equivalent to `GLiNER2.create_schema()`).
pub fn create_schema() -> Schema {
    Schema::new()
}

/// Metadata derived from a loaded [`ExtractionSchema`] (Python `batch_extract` with a schema dict).
pub fn infer_metadata_from_schema(schema: &ExtractionSchema) -> ExtractionMetadata {
    let mut meta = ExtractionMetadata {
        entity_order: schema.entities.keys().cloned().collect(),
        classification_tasks: schema
            .classifications
            .iter()
            .map(|c| c.task.clone())
            .collect(),
        ..Default::default()
    };

    for block in &schema.json_structures {
        for (parent, fmap) in &block.parents {
            meta.field_orders
                .insert(parent.clone(), fmap.keys().cloned().collect());
            for (fname, fval) in fmap {
                let key = format!("{parent}.{fname}");
                let fm = match fval {
                    StructureField::Plain(_) => FieldMeta {
                        dtype: ValueDtype::List,
                        threshold: None,
                        choices: None,
                        validators: vec![],
                    },
                    StructureField::Rich(body) => FieldMeta {
                        dtype: body.dtype.unwrap_or(ValueDtype::List),
                        threshold: body.threshold,
                        choices: body.choices.clone(),
                        validators: parse_validator_specs(&body.validators),
                    },
                };
                meta.field_metadata.insert(key, fm);
            }
        }
    }

    for block in &schema.relations {
        for rel_name in block.types.keys() {
            if !meta.relation_order.contains(rel_name) {
                meta.relation_order.push(rel_name.clone());
            }
            meta.field_orders
                .insert(rel_name.clone(), vec!["head".into(), "tail".into()]);
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
    /// Extract structured task information from a schema document.
    pub fn from_schema(schema: &ExtractionSchema) -> Self {
        let mut info = SchemaInfo::default();

        for name in schema.entities.keys() {
            let description = schema
                .entity_descriptions
                .get(name)
                .filter(|s| !s.is_empty())
                .cloned();
            info.entities.push(EntityTypeInfo {
                name: name.clone(),
                description,
            });
        }

        for block in &schema.relations {
            for rel_name in block.types.keys() {
                info.relations.push(RelationTypeInfo {
                    name: rel_name.clone(),
                });
            }
        }

        for cls in &schema.classifications {
            if !cls.task.is_empty() {
                info.classifications.push(ClassificationTaskInfo {
                    task: cls.task.clone(),
                    labels: cls.labels.clone(),
                    multi_label: cls.multi_label,
                });
            }
        }

        for st in &schema.json_structures {
            for (parent, fmap) in &st.parents {
                let mut fields = Vec::new();
                for (fname, fval) in fmap {
                    let (dtype, choices) = match fval {
                        StructureField::Plain(_) => (ValueDtype::List, None),
                        StructureField::Rich(body) => {
                            let dt = body.dtype.unwrap_or(ValueDtype::List);
                            (dt, body.choices.clone())
                        }
                    };
                    fields.push(StructureFieldInfo {
                        name: fname.clone(),
                        dtype,
                        choices,
                    });
                }
                info.structures.push(StructureInfo {
                    name: parent.clone(),
                    fields,
                });
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
    fn schema_document_alias_matches_extraction_schema() {
        let doc: SchemaDocument = ExtractionSchema::default();
        let _: ExtractionSchema = doc;
    }

    #[test]
    fn infer_metadata_entity_order() {
        let v: ExtractionSchema = serde_json::from_value(json!({
            "entities": { "a": "", "b": "" },
            "classifications": [],
            "json_structures": [],
            "relations": []
        }))
        .unwrap();
        let m = infer_metadata_from_schema(&v);
        assert_eq!(m.entity_order, vec!["a", "b"]);
    }

    #[test]
    fn parse_field_restaurant() {
        let p = parse_field_spec(&FieldSpecSource::Str(
            "restaurant::str::Restaurant name".into(),
        ))
        .unwrap();
        assert_eq!(p.name, "restaurant");
        assert_eq!(p.dtype, ValueDtype::Str);
        assert!(p.choices.is_none());
        assert_eq!(p.description.as_deref(), Some("Restaurant name"));
    }

    #[test]
    fn parse_field_seating() {
        let p = parse_field_spec(&FieldSpecSource::Str(
            "seating::[indoor|outdoor|bar]::Seating preference".into(),
        ))
        .unwrap();
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
        let p = parse_field_spec(&FieldSpecSource::Str(
            "dietary::[vegetarian|vegan|gluten-free|none]::list::Dietary restrictions".into(),
        ))
        .unwrap();
        assert_eq!(p.name, "dietary");
        assert_eq!(p.dtype, ValueDtype::List);
        assert_eq!(p.choices.as_ref().unwrap().len(), 4);
        assert_eq!(p.description.as_deref(), Some("Dietary restrictions"));
    }

    #[test]
    fn parse_field_dict() {
        let p = parse_field_spec(&FieldSpecSource::Obj(FieldSpecObject {
            name: "x".into(),
            dtype: Some("str".into()),
            choices: Some(vec!["a".into(), "b".into()]),
            description: Some("d".into()),
            validators: None,
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
        let m: IndexMap<String, Vec<FieldSpecSource>> = serde_json::from_value(json!({
            "product_info": [
                "name::str",
                "price::str",
                "features::list",
                "availability::str::[in_stock|pre_order|sold_out]"
            ]
        }))
        .unwrap();
        s.extract_json_structures(&m).unwrap();
        let (v, meta) = s.build();
        assert_eq!(v.json_structures.len(), 1);
        let fields = v.json_structures[0]
            .parents
            .get("product_info")
            .expect("product_info");
        assert!(fields.contains_key("name"));
        let av = fields.get("availability").expect("availability");
        assert!(matches!(av, StructureField::Rich(b) if b.choices.is_some()));
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
        let v: ExtractionSchema = serde_json::from_value(json!({
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
        }))
        .unwrap();
        let m = infer_metadata_from_schema(&v);
        let fm = m.field_metadata.get("form.code").expect("form.code");
        assert_eq!(fm.validators.len(), 1);
        assert!(fm.validators[0].validate("042"));
        assert!(!fm.validators[0].validate("42"));
    }

    #[test]
    fn parse_field_spec_with_validators() {
        let p = parse_field_spec(&FieldSpecSource::Obj(FieldSpecObject {
            name: "email".into(),
            dtype: Some("str".into()),
            choices: None,
            description: None,
            validators: Some(vec![ValidatorSpec {
                pattern: "@".into(),
                mode: Some("partial".into()),
                exclude: None,
                case_insensitive: None,
            }]),
        }))
        .unwrap();
        assert_eq!(p.name, "email");
        assert_eq!(p.validators.len(), 1);
        assert!(p.validators[0].validate("a@b"));
    }

    #[test]
    fn introspect_entities_with_descriptions() {
        let mut s = Schema::new();
        let mut m = IndexMap::new();
        m.insert(
            "person".into(),
            EntityTypeConfigInput::DescriptionOnly("A person's name".into()),
        );
        m.insert(
            "org".into(),
            EntityTypeConfigInput::DescriptionOnly("An organization".into()),
        );
        s.entities(EntityTypesInput::WithMeta(m));
        let (v, _) = s.build();
        let info = SchemaInfo::from_schema(&v);
        assert_eq!(info.entities.len(), 2);
        let person = info.entities.iter().find(|e| e.name == "person").unwrap();
        assert_eq!(person.description.as_deref(), Some("A person's name"));
    }

    #[test]
    fn introspect_entities_simple() {
        let mut s = Schema::new();
        s.entities(EntityTypesInput::Many(vec!["person".into(), "org".into()]));
        let (v, _) = s.build();
        let info = SchemaInfo::from_schema(&v);
        assert_eq!(info.entities.len(), 2);
        assert!(info.entities[0].description.is_none());
    }

    #[test]
    fn introspect_relations() {
        let mut s = Schema::new();
        s.relations(RelationTypesInput::Many(vec![
            "works_for".into(),
            "located_in".into(),
        ]));
        let (v, _) = s.build();
        let info = SchemaInfo::from_schema(&v);
        assert_eq!(info.relations.len(), 2);
        assert_eq!(info.relations[0].name, "works_for");
        assert_eq!(info.relations[1].name, "located_in");
    }

    #[test]
    fn introspect_classifications() {
        let mut s = Schema::new();
        s.classification(
            "sentiment",
            ClassificationLabelsInput::List(vec!["positive".into(), "negative".into()]),
            false,
            0.5,
        );
        s.classification(
            "topic",
            ClassificationLabelsInput::List(vec![
                "tech".into(),
                "sports".into(),
                "politics".into(),
            ]),
            true,
            0.3,
        );
        let (v, _) = s.build();
        let info = SchemaInfo::from_schema(&v);
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
        let m: IndexMap<String, Vec<FieldSpecSource>> = serde_json::from_value(json!({
            "product_info": [
                "name::str",
                "features::list",
                "availability::str::[in_stock|pre_order|sold_out]"
            ]
        }))
        .unwrap();
        s.extract_json_structures(&m).unwrap();
        let (v, _) = s.build();
        let info = SchemaInfo::from_schema(&v);
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
        s.entities(EntityTypesInput::Many(vec!["person".into()]));
        s.relations(RelationTypesInput::Many(vec!["works_for".into()]));
        s.classification_simple(
            "sentiment",
            ClassificationLabelsInput::List(vec!["pos".into(), "neg".into()]),
        );
        let (v, _) = s.build();
        let info = SchemaInfo::from_schema(&v);
        assert_eq!(info.entities.len(), 1);
        assert_eq!(info.relations.len(), 1);
        assert_eq!(info.classifications.len(), 1);
        assert!(info.structures.is_empty());
    }
}
