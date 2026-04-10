//! Typed extraction payloads.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Top-level extraction object (one sample).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExtractionOutput {
    #[serde(flatten)]
    pub fields: BTreeMap<String, TaskValue>,
}

/// Any value stored under a task name before or after `format_results`.
///
/// `untagged` matches the JSON shapes produced by the decoder (`null`, scalars, arrays, objects).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TaskValue {
    Null,
    String(String),
    StringArray(Vec<String>),
    /// JSON array with heterogeneous elements (relations rows, span lists, etc.).
    Array(Vec<TaskValue>),
    ObjectArray(Vec<BTreeMap<String, TaskValue>>),
    Object(BTreeMap<String, TaskValue>),
    LabelConfidence(LabelConfidence),
    LabelConfidenceList(Vec<LabelConfidence>),
    /// Listed before `F64` so JSON integers round-trip as integers (Label Studio and similar use `as_u64()`).
    U64(u64),
    F64(f64),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LabelConfidence {
    pub label: String,
    pub confidence: f32,
}

impl TaskValue {
    pub fn is_nullish(&self) -> bool {
        matches!(self, TaskValue::Null)
    }

    pub fn object_array_is_empty(&self) -> bool {
        matches!(self, TaskValue::ObjectArray(a) if a.is_empty())
    }

    pub fn array_is_empty(&self) -> bool {
        matches!(self, TaskValue::Array(a) if a.is_empty())
    }
}
