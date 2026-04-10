//! Label Studio integration (CLI-only, not part of the library crate).
//!
//! Generates Label Studio XML configs from a [`SchemaInfo`], converts gliner2
//! extraction results to Label Studio pre-annotation tasks, and provides
//! API helpers to create projects and import tasks.

use anyhow::{Context, Result};
use gliner2::schema::{SchemaInfo, StructureFieldInfo, StructureInfo, ValueDtype};
use serde::Serialize;
use serde_json::{Value, json};

// ---------------------------------------------------------------------------
// Color palette for label backgrounds
// ---------------------------------------------------------------------------

const COLORS: &[&str] = &[
    "#ff0000", "#1e90ff", "#2ecc71", "#ff00ff", "#ff8c00", "#00ced1", "#9b59b6", "#e67e22",
    "#3498db", "#e74c3c", "#1abc9c", "#f39c12", "#8e44ad", "#16a085", "#d35400", "#2980b9",
    "#c0392b", "#27ae60", "#7f8c8d", "#34495e",
];

fn color_for(index: usize) -> &'static str {
    COLORS[index % COLORS.len()]
}

// ---------------------------------------------------------------------------
// Label Studio XML config types (serialized via quick-xml + serde)
// ---------------------------------------------------------------------------

/// Root `<View>` element of a Label Studio label config.
#[derive(Serialize)]
#[serde(rename = "View")]
pub struct LabelConfig {
    #[serde(rename = "$value")]
    pub children: Vec<LabelConfigElement>,
}

/// One element inside a `<View>`.
#[derive(Serialize)]
pub enum LabelConfigElement {
    Labels(Labels),
    Relations(Relations),
    Choices(Choices),
    Text(Text),
}

/// `<Labels name="..." toName="...">` containing `<Label/>` children.
#[derive(Serialize)]
pub struct Labels {
    #[serde(rename = "@name")]
    pub name: String,
    #[serde(rename = "@toName")]
    pub to_name: String,
    #[serde(rename = "$value")]
    pub labels: Vec<Label>,
}

/// `<Label value="..." background="..."/>` (self-closing).
#[derive(Serialize)]
pub struct Label {
    #[serde(rename = "@value")]
    pub value: String,
    #[serde(rename = "@background")]
    pub background: String,
}

/// `<Relations>` containing `<Relation/>` children.
#[derive(Serialize)]
pub struct Relations {
    #[serde(rename = "$value")]
    pub relations: Vec<Relation>,
}

/// `<Relation value="..."/>` (self-closing).
#[derive(Serialize)]
pub struct Relation {
    #[serde(rename = "@value")]
    pub value: String,
}

/// `<Choices name="..." toName="..." choice="..." showInline="...">` with `<Choice/>` children.
#[derive(Serialize)]
pub struct Choices {
    #[serde(rename = "@name")]
    pub name: String,
    #[serde(rename = "@toName")]
    pub to_name: String,
    #[serde(rename = "@choice")]
    pub choice: String,
    #[serde(rename = "@showInline")]
    pub show_inline: String,
    #[serde(rename = "$value")]
    pub choices: Vec<Choice>,
}

/// `<Choice value="..."/>` (self-closing).
#[derive(Serialize)]
pub struct Choice {
    #[serde(rename = "@value")]
    pub value: String,
}

/// `<Text name="..." value="..."/>` (self-closing).
#[derive(Serialize)]
pub struct Text {
    #[serde(rename = "@name")]
    pub name: String,
    #[serde(rename = "@value")]
    pub value: String,
}

// ---------------------------------------------------------------------------
// Label config generation
// ---------------------------------------------------------------------------

impl LabelConfig {
    /// Build a Label Studio label config from schema info.
    pub fn from_schema(info: &SchemaInfo) -> Self {
        let mut children = Vec::new();
        let mut color_idx = 0usize;

        // Entities -> <Labels>
        if !info.entities.is_empty() {
            let labels = info
                .entities
                .iter()
                .map(|ent| {
                    let l = Label {
                        value: ent.name.clone(),
                        background: color_for(color_idx).into(),
                    };
                    color_idx += 1;
                    l
                })
                .collect();
            children.push(LabelConfigElement::Labels(Labels {
                name: "label".into(),
                to_name: "text".into(),
                labels,
            }));
        }

        // Relations -> <Relations> + optional generic span labels
        if !info.relations.is_empty() {
            children.push(LabelConfigElement::Relations(Relations {
                relations: info
                    .relations
                    .iter()
                    .map(|r| Relation {
                        value: r.name.clone(),
                    })
                    .collect(),
            }));

            // If there are no entity labels, add generic relation span labels
            if info.entities.is_empty() {
                let head_label = Label {
                    value: "head".into(),
                    background: color_for(color_idx).into(),
                };
                color_idx += 1;
                let tail_label = Label {
                    value: "tail".into(),
                    background: color_for(color_idx).into(),
                };
                color_idx += 1;
                children.push(LabelConfigElement::Labels(Labels {
                    name: "relation_spans".into(),
                    to_name: "text".into(),
                    labels: vec![head_label, tail_label],
                }));
            }
        }

        // Classifications -> <Choices>
        for cls in &info.classifications {
            children.push(LabelConfigElement::Choices(Choices {
                name: cls.task.clone(),
                to_name: "text".into(),
                choice: if cls.multi_label {
                    "multiple"
                } else {
                    "single"
                }
                .into(),
                show_inline: "true".into(),
                choices: cls
                    .labels
                    .iter()
                    .map(|l| Choice { value: l.clone() })
                    .collect(),
            }));
        }

        // Structures -> <Labels> for span fields, <Choices> for choice fields
        for st in &info.structures {
            let span_fields: Vec<&StructureFieldInfo> =
                st.fields.iter().filter(|f| f.choices.is_none()).collect();
            let choice_fields: Vec<&StructureFieldInfo> =
                st.fields.iter().filter(|f| f.choices.is_some()).collect();

            if !span_fields.is_empty() {
                let labels = span_fields
                    .iter()
                    .map(|f| {
                        let l = Label {
                            value: f.name.clone(),
                            background: color_for(color_idx).into(),
                        };
                        color_idx += 1;
                        l
                    })
                    .collect();
                children.push(LabelConfigElement::Labels(Labels {
                    name: st.name.clone(),
                    to_name: "text".into(),
                    labels,
                }));
            }

            for f in &choice_fields {
                children.push(LabelConfigElement::Choices(Choices {
                    name: format!("{}__{}", st.name, f.name),
                    to_name: "text".into(),
                    choice: if f.dtype == ValueDtype::List {
                        "multiple"
                    } else {
                        "single"
                    }
                    .into(),
                    show_inline: "true".into(),
                    choices: f
                        .choices
                        .as_ref()
                        .unwrap()
                        .iter()
                        .map(|c| Choice { value: c.clone() })
                        .collect(),
                }));
            }
        }

        // Text object (always present, exactly once)
        children.push(LabelConfigElement::Text(Text {
            name: "text".into(),
            value: "$text".into(),
        }));

        LabelConfig { children }
    }

    /// Serialize to Label Studio XML string.
    pub fn to_xml(&self) -> String {
        quick_xml::se::to_string(self).expect("LabelConfig serialization cannot fail")
    }
}

/// Generate a Label Studio XML label config string from schema info.
pub fn generate_label_config(info: &SchemaInfo) -> String {
    LabelConfig::from_schema(info).to_xml()
}

// ---------------------------------------------------------------------------
// Result -> Label Studio task conversion
// ---------------------------------------------------------------------------

/// Convert a single gliner2 extraction result to a Label Studio task with pre-annotations.
///
/// Expects results produced with `include_spans=true` and `include_confidence=true`.
pub fn convert_result_to_task(text: &str, result: &Value, info: &SchemaInfo) -> Value {
    let mut annotations: Vec<Value> = Vec::new();

    if let Some(result_obj) = result.as_object() {
        // Entities
        if let Some(entities) = result_obj.get("entities").and_then(|v| v.as_object()) {
            convert_entities(&mut annotations, entities, "label");
        }

        // Relations
        if let Some(rel_ext) = result_obj
            .get("relation_extraction")
            .and_then(|v| v.as_object())
        {
            let labels_name = if info.entities.is_empty() {
                "relation_spans"
            } else {
                "label"
            };
            convert_relations(&mut annotations, rel_ext, labels_name);
        }

        // Classifications
        for cls in &info.classifications {
            if let Some(val) = result_obj.get(&cls.task) {
                convert_classification(&mut annotations, &cls.task, val);
            }
        }

        // Structures
        for st in &info.structures {
            if let Some(val) = result_obj.get(&st.name) {
                convert_structure(&mut annotations, st, val);
            }
        }
    }

    json!({
        "data": { "text": text },
        "predictions": [{
            "model_version": "gliner2",
            "result": annotations,
        }]
    })
}

fn convert_entities(
    annotations: &mut Vec<Value>,
    entities: &serde_json::Map<String, Value>,
    labels_name: &str,
) {
    for (label, items) in entities {
        if let Some(arr) = items.as_array() {
            for item in arr {
                if let Some(ann) = span_to_label_annotation(item, label, labels_name) {
                    annotations.push(ann);
                }
            }
        }
    }
}

fn span_to_label_annotation(item: &Value, label: &str, from_name: &str) -> Option<Value> {
    let obj = item.as_object()?;
    let start = obj.get("start")?.as_u64()?;
    let end = obj.get("end")?.as_u64()?;
    let text = obj.get("text")?.as_str()?;
    let id = format!("{}_{start}_{end}", from_name);
    let confidence = obj.get("confidence").and_then(|v| v.as_f64());

    let mut ann = json!({
        "id": id,
        "from_name": from_name,
        "to_name": "text",
        "type": "labels",
        "value": {
            "start": start,
            "end": end,
            "text": text,
            "labels": [label],
        }
    });

    if let Some(score) = confidence {
        ann["score"] = json!(score);
    }

    Some(ann)
}

fn convert_relations(
    annotations: &mut Vec<Value>,
    rel_ext: &serde_json::Map<String, Value>,
    labels_name: &str,
) {
    for (rel_name, items) in rel_ext {
        if let Some(arr) = items.as_array() {
            for (i, item) in arr.iter().enumerate() {
                let obj = match item.as_object() {
                    Some(o) => o,
                    None => continue,
                };
                let head = match obj.get("head") {
                    Some(v) => v,
                    None => continue,
                };
                let tail = match obj.get("tail") {
                    Some(v) => v,
                    None => continue,
                };

                let head_id = format!("rel_{rel_name}_{i}_head");
                let tail_id = format!("rel_{rel_name}_{i}_tail");

                // Create span regions for head and tail
                if let Some(mut head_ann) = span_to_label_annotation(head, "head", labels_name) {
                    head_ann["id"] = json!(head_id.clone());
                    annotations.push(head_ann);
                }
                if let Some(mut tail_ann) = span_to_label_annotation(tail, "tail", labels_name) {
                    tail_ann["id"] = json!(tail_id.clone());
                    annotations.push(tail_ann);
                }

                // Create the relation linking them
                annotations.push(json!({
                    "from_id": head_id,
                    "to_id": tail_id,
                    "type": "relation",
                    "direction": "right",
                    "labels": [rel_name],
                }));
            }
        }
    }
}

fn convert_classification(annotations: &mut Vec<Value>, task: &str, val: &Value) {
    let choices: Vec<String> = match val {
        Value::String(s) => vec![s.clone()],
        Value::Array(arr) => arr
            .iter()
            .filter_map(|v| match v {
                Value::String(s) => Some(s.clone()),
                // With include_confidence, labels may be objects
                Value::Object(o) => o.get("label").and_then(|l| l.as_str()).map(String::from),
                _ => None,
            })
            .collect(),
        Value::Object(o) => {
            // Single-label with confidence: {"label": "X", "confidence": 0.9}
            o.get("label")
                .and_then(|l| l.as_str())
                .map(|s| vec![s.to_string()])
                .unwrap_or_default()
        }
        _ => return,
    };

    if !choices.is_empty() {
        annotations.push(json!({
            "from_name": task,
            "to_name": "text",
            "type": "choices",
            "value": { "choices": choices },
        }));
    }
}

fn convert_structure(annotations: &mut Vec<Value>, st: &StructureInfo, val: &Value) {
    let instances = match val {
        Value::Array(arr) => arr.as_slice(),
        _ => return,
    };

    for instance in instances {
        let obj = match instance.as_object() {
            Some(o) => o,
            None => continue,
        };

        for field in &st.fields {
            let fval = match obj.get(&field.name) {
                Some(v) => v,
                None => continue,
            };

            if field.choices.is_some() {
                // Choice field -> Choices annotation
                let choices: Vec<String> = match fval {
                    Value::String(s) => vec![s.clone()],
                    Value::Array(arr) => arr
                        .iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect(),
                    _ => continue,
                };
                if !choices.is_empty() {
                    let name = format!("{}__{}", st.name, field.name);
                    annotations.push(json!({
                        "from_name": name,
                        "to_name": "text",
                        "type": "choices",
                        "value": { "choices": choices },
                    }));
                }
            } else {
                // Span field -> Labels annotation
                let items: Vec<&Value> = match fval {
                    Value::Array(arr) => arr.iter().collect(),
                    other => vec![other],
                };
                for item in items {
                    if let Some(ann) = span_to_label_annotation(item, &field.name, &st.name) {
                        annotations.push(ann);
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Label Studio API
// ---------------------------------------------------------------------------

/// Find an existing project by title, or create a new one with the given label config.
pub fn create_or_get_project(url: &str, api_key: &str, name: &str, config: &str) -> Result<u64> {
    let base = url.trim_end_matches('/');

    // Search for existing project with matching title
    let list_url = format!("{base}/api/projects");
    let resp: Value = ureq::get(&list_url)
        .header("Authorization", &format!("Token {api_key}"))
        .call()
        .context("Failed to list Label Studio projects")?
        .body_mut()
        .read_json()
        .context("Failed to parse project list response")?;

    // Response may be paginated: { "results": [...] } or just an array
    let results = resp
        .get("results")
        .and_then(|v| v.as_array())
        .or_else(|| resp.as_array());

    if let Some(projects) = results {
        for proj in projects {
            if proj.get("title").and_then(|t| t.as_str()) == Some(name) {
                if let Some(id) = proj.get("id").and_then(|v| v.as_u64()) {
                    tracing::info!(project_id = id, "Found existing Label Studio project");
                    return Ok(id);
                }
            }
        }
    }

    // Create new project
    let create_url = format!("{base}/api/projects");
    let body = json!({
        "title": name,
        "label_config": config,
    });

    let resp: Value = ureq::post(&create_url)
        .header("Authorization", &format!("Token {api_key}"))
        .header("Content-Type", "application/json")
        .send_json(&body)
        .context("Failed to create Label Studio project")?
        .body_mut()
        .read_json()
        .context("Failed to parse project creation response")?;

    let id = resp
        .get("id")
        .and_then(|v| v.as_u64())
        .context("Label Studio project creation response missing 'id'")?;

    tracing::info!(
        project_id = id,
        title = name,
        "Created new Label Studio project"
    );
    Ok(id)
}

/// Import tasks (with pre-annotations) into a Label Studio project.
///
/// Chunks into batches of 100 to stay within API limits.
pub fn import_tasks(url: &str, api_key: &str, project_id: u64, tasks: &[Value]) -> Result<()> {
    if tasks.is_empty() {
        return Ok(());
    }

    let base = url.trim_end_matches('/');
    let import_url = format!("{base}/api/projects/{project_id}/import");

    for (chunk_idx, chunk) in tasks.chunks(100).enumerate() {
        let body = Value::Array(chunk.to_vec());

        ureq::post(&import_url)
            .header("Authorization", &format!("Token {api_key}"))
            .header("Content-Type", "application/json")
            .send_json(&body)
            .with_context(|| {
                format!(
                    "Failed to import tasks chunk {} ({} tasks) to project {project_id}",
                    chunk_idx + 1,
                    chunk.len(),
                )
            })?;

        tracing::debug!(
            chunk = chunk_idx + 1,
            tasks = chunk.len(),
            "Imported task chunk"
        );
    }

    tracing::info!(
        total_tasks = tasks.len(),
        project_id,
        "Successfully imported all tasks to Label Studio"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use gliner2::schema::{
        ClassificationTaskInfo, EntityTypeInfo, RelationTypeInfo, StructureFieldInfo, StructureInfo,
    };

    fn sample_entity_info() -> SchemaInfo {
        SchemaInfo {
            entities: vec![
                EntityTypeInfo {
                    name: "person".into(),
                    description: Some("A person".into()),
                },
                EntityTypeInfo {
                    name: "org".into(),
                    description: None,
                },
            ],
            ..Default::default()
        }
    }

    #[test]
    fn label_config_entities() {
        let info = sample_entity_info();
        let config = generate_label_config(&info);
        assert!(config.contains("<Labels name=\"label\" toName=\"text\">"));
        assert!(config.contains("<Label value=\"person\""));
        assert!(config.contains("<Label value=\"org\""));
        assert!(config.contains("<Text name=\"text\" value=\"$text\"/>"));
        assert!(config.starts_with("<View>"));
        assert!(config.ends_with("</View>"));
    }

    #[test]
    fn label_config_entities_struct() {
        let info = sample_entity_info();
        let lc = LabelConfig::from_schema(&info);
        assert_eq!(lc.children.len(), 2); // Labels + Text
        match &lc.children[0] {
            LabelConfigElement::Labels(labels) => {
                assert_eq!(labels.name, "label");
                assert_eq!(labels.labels.len(), 2);
                assert_eq!(labels.labels[0].value, "person");
            }
            _ => panic!("expected Labels"),
        }
    }

    #[test]
    fn label_config_relations_without_entities() {
        let info = SchemaInfo {
            relations: vec![RelationTypeInfo {
                name: "works_for".into(),
            }],
            ..Default::default()
        };
        let config = generate_label_config(&info);
        assert!(config.contains("<Relations>"));
        assert!(config.contains("<Relation value=\"works_for\"/>"));
        assert!(config.contains("<Labels name=\"relation_spans\""));
        assert!(config.contains("<Label value=\"head\""));
        assert!(config.contains("<Label value=\"tail\""));
    }

    #[test]
    fn label_config_relations_with_entities() {
        let info = SchemaInfo {
            entities: vec![EntityTypeInfo {
                name: "person".into(),
                description: None,
            }],
            relations: vec![RelationTypeInfo {
                name: "works_for".into(),
            }],
            ..Default::default()
        };
        let config = generate_label_config(&info);
        assert!(config.contains("<Relations>"));
        // Should NOT have separate relation_spans labels when entities exist
        assert!(!config.contains("relation_spans"));
    }

    #[test]
    fn label_config_classifications() {
        let info = SchemaInfo {
            classifications: vec![
                ClassificationTaskInfo {
                    task: "sentiment".into(),
                    labels: vec!["positive".into(), "negative".into()],
                    multi_label: false,
                },
                ClassificationTaskInfo {
                    task: "topics".into(),
                    labels: vec!["tech".into(), "sports".into()],
                    multi_label: true,
                },
            ],
            ..Default::default()
        };
        let config = generate_label_config(&info);
        assert!(config.contains("choice=\"single\""));
        assert!(config.contains("choice=\"multiple\""));
        assert!(config.contains("<Choice value=\"positive\"/>"));
        assert!(config.contains("name=\"sentiment\""));
        assert!(config.contains("name=\"topics\""));
    }

    #[test]
    fn label_config_structures() {
        let info = SchemaInfo {
            structures: vec![StructureInfo {
                name: "product".into(),
                fields: vec![
                    StructureFieldInfo {
                        name: "name".into(),
                        dtype: ValueDtype::Str,
                        choices: None,
                    },
                    StructureFieldInfo {
                        name: "status".into(),
                        dtype: ValueDtype::Str,
                        choices: Some(vec!["active".into(), "discontinued".into()]),
                    },
                ],
            }],
            ..Default::default()
        };
        let config = generate_label_config(&info);
        // Span field -> Labels
        assert!(config.contains("<Labels name=\"product\""));
        assert!(config.contains("<Label value=\"name\""));
        // Choice field -> Choices
        assert!(config.contains("name=\"product__status\""));
        assert!(config.contains("<Choice value=\"active\"/>"));
    }

    #[test]
    fn convert_entity_result() {
        let info = sample_entity_info();
        let result = json!({
            "entities": {
                "person": [
                    {"text": "John", "start": 0, "end": 4, "confidence": 0.95}
                ],
                "org": [
                    {"text": "Acme", "start": 10, "end": 14, "confidence": 0.88}
                ]
            }
        });
        let task = convert_result_to_task("John works at Acme", &result, &info);
        assert_eq!(task["data"]["text"], "John works at Acme");
        let preds = task["predictions"].as_array().unwrap();
        assert_eq!(preds.len(), 1);
        let results = preds[0]["result"].as_array().unwrap();
        assert_eq!(results.len(), 2);
        // Find the person entity annotation (order may vary)
        let ann = results
            .iter()
            .find(|a| a["value"]["labels"][0] == "person")
            .expect("person annotation");
        assert_eq!(ann["type"], "labels");
        assert_eq!(ann["from_name"], "label");
        assert_eq!(ann["value"]["start"], 0);
        assert_eq!(ann["value"]["end"], 4);
    }

    #[test]
    fn convert_classification_result() {
        let info = SchemaInfo {
            classifications: vec![ClassificationTaskInfo {
                task: "sentiment".into(),
                labels: vec!["positive".into(), "negative".into()],
                multi_label: false,
            }],
            ..Default::default()
        };
        let result = json!({ "sentiment": "positive" });
        let task = convert_result_to_task("Great product!", &result, &info);
        let results = task["predictions"][0]["result"].as_array().unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["type"], "choices");
        assert_eq!(results[0]["value"]["choices"][0], "positive");
    }

    #[test]
    fn convert_relation_result() {
        let info = SchemaInfo {
            relations: vec![RelationTypeInfo {
                name: "works_for".into(),
            }],
            ..Default::default()
        };
        let result = json!({
            "relation_extraction": {
                "works_for": [{
                    "head": {"text": "John", "start": 0, "end": 4, "confidence": 0.9},
                    "tail": {"text": "Acme", "start": 14, "end": 18, "confidence": 0.85}
                }]
            }
        });
        let task = convert_result_to_task("John works for Acme", &result, &info);
        let results = task["predictions"][0]["result"].as_array().unwrap();
        // head region + tail region + relation = 3 annotations
        assert_eq!(results.len(), 3);
        assert_eq!(results[2]["type"], "relation");
        assert_eq!(results[2]["labels"][0], "works_for");
    }
}
