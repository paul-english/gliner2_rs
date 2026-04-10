//! Typed wire-format schema (Python `GLiNER2` schema dict). Key order matches JSON / Python 3.7+.

use super::ValueDtype;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

/// Root multitask schema passed to preprocessing and extraction.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ExtractionSchema {
    #[serde(default, skip_serializing_if = "IndexMap::is_empty")]
    pub entities: IndexMap<String, String>,
    #[serde(default, skip_serializing_if = "IndexMap::is_empty")]
    pub entity_descriptions: IndexMap<String, String>,
    #[serde(default, skip_serializing_if = "IndexMap::is_empty")]
    pub json_descriptions: IndexMap<String, IndexMap<String, String>>,
    #[serde(default)]
    pub json_structures: Vec<JsonStructureBlock>,
    #[serde(default)]
    pub classifications: Vec<ClassificationConfig>,
    #[serde(default)]
    pub relations: Vec<RelationBlock>,
}

impl Default for ExtractionSchema {
    fn default() -> Self {
        Self {
            entities: IndexMap::new(),
            entity_descriptions: IndexMap::new(),
            json_descriptions: IndexMap::new(),
            json_structures: Vec::new(),
            classifications: Vec::new(),
            relations: Vec::new(),
        }
    }
}

/// One element of `json_structures`: `{ parent: { field: spec, ... } }` (one or more parents).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct JsonStructureBlock {
    #[serde(flatten)]
    pub parents: IndexMap<String, IndexMap<String, StructureField>>,
}

/// Field definition under a JSON structure parent.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum StructureField {
    /// List-style field encoded as a string (usually `""`).
    Plain(String),
    Rich(StructureFieldBody),
}

impl StructureField {
    pub fn as_body_mut(&mut self) -> Option<&mut StructureFieldBody> {
        match self {
            StructureField::Plain(_) => None,
            StructureField::Rich(b) => Some(b),
        }
    }

    pub fn is_choice_field(&self) -> bool {
        match self {
            StructureField::Plain(_) => false,
            StructureField::Rich(b) => b.value.is_some() && b.choices.is_some(),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct StructureFieldBody {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<FieldDefault>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub choices: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dtype: Option<ValueDtype>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threshold: Option<f32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub validators: Vec<ValidatorSpec>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum FieldDefault {
    Str(String),
    Arr(Vec<String>),
}

/// Regex validator entry in a schema field (mirrors Python).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ValidatorSpec {
    pub pattern: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exclude: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub case_insensitive: Option<bool>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ClassificationConfig {
    pub task: String,
    pub labels: Vec<String>,
    #[serde(default)]
    pub multi_label: bool,
    #[serde(default = "default_cls_threshold")]
    pub cls_threshold: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub true_label: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label_descriptions: Option<IndexMap<String, String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub examples: Option<Vec<ClassificationExample>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub class_act: Option<String>,
}

fn default_cls_threshold() -> f32 {
    0.5
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum ClassificationExample {
    /// At least two strings; extra elements are ignored when converting to `(input, output)`.
    List(Vec<String>),
    Obj(ClassificationExampleObj),
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ClassificationExampleObj {
    pub input: String,
    pub output: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RelationBlock {
    #[serde(flatten)]
    pub types: IndexMap<String, RelationEndpoints>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RelationEndpoints {
    pub head: String,
    pub tail: String,
}

// ---------------------------------------------------------------------------
// Builder / CLI input shapes (not necessarily the full wire document)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EntityTypesInput {
    One(String),
    Many(Vec<String>),
    WithMeta(IndexMap<String, EntityTypeConfigInput>),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EntityTypeConfigInput {
    DescriptionOnly(String),
    Full(EntityTypeConfig),
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct EntityTypeConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dtype: Option<ValueDtype>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threshold: Option<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ClassificationLabelsInput {
    List(Vec<String>),
    WithDescriptions(IndexMap<String, String>),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RelationTypesInput {
    One(String),
    Many(Vec<String>),
    WithMeta(IndexMap<String, RelationTypeConfigInput>),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RelationTypeConfigInput {
    DescriptionOnly(String),
    Full(RelationTypeConfig),
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RelationTypeConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threshold: Option<f32>,
}

/// `extract_json` field spec: string DSL or JSON object.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FieldSpecSource {
    Str(String),
    Obj(FieldSpecObject),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FieldSpecObject {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dtype: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub choices: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub validators: Option<Vec<ValidatorSpec>>,
}

/// Multitask schema in wire/JSON shape (alias for [`ExtractionSchema`]).
pub type SchemaDocument = ExtractionSchema;
