//! Multi-task input formatting aligned with `gliner2.processor.SchemaTransformer` (inference).

use crate::processor::SchemaTransformer;
use crate::schema::{ClassificationExample, ExtractionSchema, FieldDefault, StructureField};
use anyhow::Result;
use indexmap::IndexMap;

pub const SEP_STRUCT: &str = "[SEP_STRUCT]";
pub const DESC_TOKEN: &str = "[DESCRIPTION]";
pub const EXAMPLE_TOKEN: &str = "[EXAMPLE]";
pub const OUTPUT_TOKEN: &str = "[OUTPUT]";

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TaskType {
    JsonStructures,
    Entities,
    Relations,
    Classifications,
}

impl TaskType {
    pub fn as_str(&self) -> &'static str {
        match self {
            TaskType::JsonStructures => "json_structures",
            TaskType::Entities => "entities",
            TaskType::Relations => "relations",
            TaskType::Classifications => "classifications",
        }
    }
}

/// Padded batch of [`PreprocessedInput`] for a single encoder forward pass.
///
/// `input_ids` and `attention_mask` are row-major `[batch_size * max_seq_len]`.
/// Padding uses id `0`, matching Python `torch.zeros` collate.
#[derive(Debug, Clone)]
pub struct PreprocessedBatch {
    pub batch_size: usize,
    pub max_seq_len: usize,
    pub input_ids: Vec<u32>,
    /// `1` = real token, `0` = pad (stored as `u32` for tensor construction).
    pub attention_mask: Vec<u32>,
    /// Per-sample fields; `input_ids` here are the **unpadded** originals (batch path uses [`PreprocessedBatch::input_ids`]).
    pub samples: Vec<PreprocessedInput>,
}

/// Pad a slice of preprocessed records to `PreprocessedBatch`. Returns `None` if `samples` is empty.
pub fn collate_preprocessed(samples: &[PreprocessedInput]) -> Option<PreprocessedBatch> {
    if samples.is_empty() {
        return None;
    }
    let batch_size = samples.len();
    let max_seq_len = samples.iter().map(|s| s.input_ids.len()).max().unwrap_or(0);
    let mut input_ids = vec![0u32; batch_size * max_seq_len];
    let mut attention_mask = vec![0u32; batch_size * max_seq_len];
    for (i, s) in samples.iter().enumerate() {
        let len = s.input_ids.len();
        let row_off = i * max_seq_len;
        input_ids[row_off..row_off + len].copy_from_slice(&s.input_ids);
        for j in 0..len {
            attention_mask[row_off + j] = 1;
        }
    }
    Some(PreprocessedBatch {
        batch_size,
        max_seq_len,
        input_ids,
        attention_mask,
        samples: samples.to_vec(),
    })
}

/// Single-sample preprocessed input for `CandleExtractor::extract_preprocessed` (or `TchExtractor`).
#[derive(Debug, Clone)]
pub struct PreprocessedInput {
    pub input_ids: Vec<u32>,
    pub text_word_first_positions: Vec<usize>,
    /// Per task block: sequence indices of `[P]`, `[C]`/`[E]`/`[R]`/`[L]` specials.
    pub schema_special_indices: Vec<Vec<usize>>,
    pub schema_tokens_list: Vec<Vec<String>>,
    pub task_types: Vec<TaskType>,
    pub text_tokens: Vec<String>,
    pub start_mappings: Vec<usize>,
    pub end_mappings: Vec<usize>,
    pub original_text: String,
    pub len_prefix: usize,
    /// Schema with choice fields wrapped like Python after `_wrap_classification_fields`.
    pub schema_working: ExtractionSchema,
    /// Original schema (pre-wrap) for decoding metadata.
    pub schema_original: ExtractionSchema,
}

impl SchemaTransformer {
    /// Full multi-task transform (inference). Matches Python `collate_fn_inference` / `_transform_record`.
    pub fn transform_extract(
        &self,
        text: &str,
        schema: &ExtractionSchema,
        max_len: Option<usize>,
    ) -> Result<PreprocessedInput> {
        let mut text = text.to_string();
        if !text.is_empty() && !text.ends_with(&['.', '!', '?'][..]) {
            text.push('.');
        } else if text.is_empty() {
            text = ".".into();
        }

        let schema_original = schema.clone();
        let mut schema_working = schema.clone();

        let prefix = build_classification_prefix(&schema_working);
        if !prefix.is_empty() {
            wrap_classification_fields(&mut schema_working);
        }

        // Match Python `WhitespaceTokenSplitter(text, lower=True)`: split lowercased string.
        let text_lc = text.to_lowercase();
        let mut text_tokens = Vec::new();
        let mut start_idx_map = Vec::new();
        let mut end_idx_map = Vec::new();
        for (tkn, start, end) in self.word_splitter.split(&text_lc) {
            text_tokens.push(tkn.to_string());
            start_idx_map.push(start);
            end_idx_map.push(end);
        }

        if let Some(ml) = max_len {
            text_tokens.truncate(ml);
            start_idx_map.truncate(ml);
            end_idx_map.truncate(ml);
        }

        let len_prefix = prefix.len();
        if !prefix.is_empty() {
            let mut combined = prefix;
            combined.append(&mut text_tokens);
            text_tokens = combined;
        }

        let inferred = infer_from_json(&schema_working)?;
        let schema_tokens_list = inferred.schemas;
        let task_types = inferred.task_types;

        let format_result = self.format_input_with_mapping(&schema_tokens_list, &text_tokens)?;

        Ok(PreprocessedInput {
            input_ids: format_result.input_ids,
            text_word_first_positions: format_result.text_word_first_positions,
            schema_special_indices: format_result.schema_special_positions,
            schema_tokens_list,
            task_types,
            text_tokens,
            start_mappings: start_idx_map,
            end_mappings: end_idx_map,
            original_text: text,
            len_prefix,
            schema_working,
            schema_original,
        })
    }

    fn format_input_with_mapping(
        &self,
        schema_tokens_list: &[Vec<String>],
        text_tokens: &[String],
    ) -> Result<FormatMappingResult> {
        let mut combined: Vec<String> = Vec::new();
        for struct_tokens in schema_tokens_list {
            combined.extend(struct_tokens.iter().cloned());
            combined.push(SEP_STRUCT.to_string());
        }
        if !combined.is_empty() {
            combined.pop();
        }
        combined.push(crate::processor::SEP_TEXT.to_string());
        combined.extend(text_tokens.iter().cloned());

        let mut input_ids = Vec::new();
        let mut text_word_first_positions = Vec::new();
        let mut schema_special_positions: Vec<Vec<usize>> =
            vec![Vec::new(); schema_tokens_list.len()];

        let num_schemas = schema_tokens_list.len();
        let text_schema_idx = num_schemas;
        let mut current_schema = 0usize;
        let mut found_sep = false;
        let mut last_text_orig: Option<usize> = None;

        for (orig_idx, token) in combined.iter().enumerate() {
            let (seg_type, schema_idx) = if token.as_str() == crate::processor::SEP_TEXT {
                found_sep = true;
                ("sep", text_schema_idx)
            } else if !found_sep {
                let sch = current_schema;
                if token == SEP_STRUCT {
                    current_schema += 1;
                }
                ("schema", sch)
            } else {
                ("text", text_schema_idx)
            };

            let encoded = self
                .tokenizer
                .encode(token.as_str(), false)
                .map_err(|e| anyhow::anyhow!("Tokenization error: {}", e))?;
            let ids = encoded.get_ids();

            if seg_type == "text" && !ids.is_empty() {
                if last_text_orig != Some(orig_idx) {
                    text_word_first_positions.push(input_ids.len());
                    last_text_orig = Some(orig_idx);
                }
            } else if seg_type == "schema"
                && schema_idx < num_schemas
                && let Some(&first_id) = ids.first()
                && self.special_token_ids.contains(&first_id)
            {
                schema_special_positions[schema_idx].push(input_ids.len());
            }

            input_ids.extend(ids.iter().copied());
        }

        Ok(FormatMappingResult {
            input_ids,
            text_word_first_positions,
            schema_special_positions,
        })
    }
}

struct FormatMappingResult {
    input_ids: Vec<u32>,
    text_word_first_positions: Vec<usize>,
    schema_special_positions: Vec<Vec<usize>>,
}

struct InferredSchemas {
    schemas: Vec<Vec<String>>,
    task_types: Vec<TaskType>,
}

fn build_classification_prefix(schema: &ExtractionSchema) -> Vec<String> {
    let mut prefix_tokens: Vec<String> = Vec::new();

    for block in &schema.json_structures {
        for (parent, fmap) in &block.parents {
            let mut cls_fields: Vec<(String, Vec<String>)> = Vec::new();
            for (fname, fval) in fmap {
                if let StructureField::Rich(body) = fval {
                    if body.value.is_some() {
                        if let Some(ch) = &body.choices {
                            cls_fields.push((fname.clone(), ch.clone()));
                        }
                    }
                }
            }
            if cls_fields.is_empty() {
                continue;
            }
            let mut inner: Vec<String> = Vec::new();
            for (fname, choices) in cls_fields {
                inner.push(fname);
                inner.push("(".into());
                for (i, c) in choices.iter().enumerate() {
                    if i > 0 {
                        inner.push('|'.to_string());
                    }
                    inner.push(c.clone());
                }
                inner.extend([")".into(), ",".into()]);
            }
            if !inner.is_empty() {
                inner.pop();
                prefix_tokens.push("(".into());
                prefix_tokens.push(format!("{}:", parent));
                prefix_tokens.extend(inner);
                prefix_tokens.push(")".into());
            }
        }
    }
    prefix_tokens
}

fn wrap_classification_fields(schema: &mut ExtractionSchema) {
    for block in &mut schema.json_structures {
        for (_parent, fmap) in block.parents.iter_mut() {
            for (_fname, fval) in fmap.iter_mut() {
                if let StructureField::Rich(body) = fval {
                    if body.choices.is_none() || body.value.is_none() {
                        continue;
                    }
                    if let Some(raw) = body.value.take() {
                        let wrapped = match raw {
                            FieldDefault::Arr(a) => FieldDefault::Arr(
                                a.into_iter().map(|s| format!("[selection]{s}")).collect(),
                            ),
                            FieldDefault::Str(s) => FieldDefault::Str(format!("[selection]{s}")),
                        };
                        body.value = Some(wrapped);
                    }
                }
            }
        }
    }
}

fn infer_from_json(schema: &ExtractionSchema) -> Result<InferredSchemas> {
    let mut schemas = Vec::new();
    let mut types = Vec::new();

    process_json_structures(schema, &mut schemas, &mut types);
    process_entities(schema, &mut schemas, &mut types);
    process_relations(schema, &mut schemas, &mut types);
    process_classifications(schema, &mut schemas, &mut types)?;

    Ok(InferredSchemas {
        schemas,
        task_types: types,
    })
}

fn transform_schema(
    parent: &str,
    fields: &[String],
    child_prefix: &str,
    label_descriptions: Option<&IndexMap<String, String>>,
    examples: &[(String, String)],
    example_mode: &str,
) -> Vec<String> {
    let mut prompt_str = parent.to_string();
    if (example_mode == "descriptions" || example_mode == "both")
        && let Some(descs) = label_descriptions
    {
        let mut pairs: Vec<_> = descs
            .iter()
            .filter(|(l, _)| fields.iter().any(|f| f == *l))
            .collect();
        pairs.sort_by_key(|(k, _)| *k);
        for (label, desc) in pairs {
            prompt_str.push(' ');
            prompt_str.push_str(DESC_TOKEN);
            prompt_str.push(' ');
            prompt_str.push_str(label);
            prompt_str.push_str(": ");
            prompt_str.push_str(desc);
        }
    }
    if example_mode == "few_shot" || example_mode == "both" {
        for (inp, out) in examples {
            if !fields.iter().any(|f| f == out) {
                continue;
            }
            prompt_str.push(' ');
            prompt_str.push_str(EXAMPLE_TOKEN);
            prompt_str.push(' ');
            prompt_str.push_str(inp);
            prompt_str.push(' ');
            prompt_str.push_str(OUTPUT_TOKEN);
            prompt_str.push(' ');
            prompt_str.push_str(out);
        }
    }

    let mut tokens = vec![
        "(".to_string(),
        crate::processor::P_TOKEN.to_string(),
        prompt_str,
        "(".to_string(),
    ];
    for f in fields {
        tokens.push(child_prefix.to_string());
        tokens.push(f.clone());
    }
    tokens.push(")".to_string());
    tokens.push(")".to_string());
    tokens
}

fn process_json_structures(
    schema: &ExtractionSchema,
    schemas: &mut Vec<Vec<String>>,
    types: &mut Vec<TaskType>,
) {
    let mut groups: Vec<(String, Vec<IndexMap<String, StructureField>>)> = Vec::new();
    for block in &schema.json_structures {
        for (parent, fmap) in &block.parents {
            if let Some(i) = groups.iter().position(|(p, _)| p == parent) {
                groups[i].1.push(fmap.clone());
            } else {
                groups.push((parent.clone(), vec![fmap.clone()]));
            }
        }
    }

    for (parent, occurrences) in groups {
        let mut common: Vec<String> = Vec::new();
        for occ in &occurrences {
            for k in occ.keys() {
                if !common.contains(k) {
                    common.push(k.clone());
                }
            }
        }
        if common.is_empty() {
            continue;
        }

        let json_descs = schema.json_descriptions.get(&parent);

        let mode = if json_descs.map(|d| !d.is_empty()).unwrap_or(false) {
            "descriptions"
        } else {
            "none"
        };

        let desc_ref = json_descs;
        let toks = transform_schema(
            &parent,
            &common,
            "[C]",
            desc_ref,
            &[],
            if mode == "none" {
                "none"
            } else {
                "descriptions"
            },
        );
        schemas.push(toks);
        types.push(TaskType::JsonStructures);
    }
}

fn process_entities(
    schema: &ExtractionSchema,
    schemas: &mut Vec<Vec<String>>,
    types: &mut Vec<TaskType>,
) {
    if schema.entities.is_empty() {
        return;
    }
    let chosen: Vec<String> = schema.entities.keys().cloned().collect();
    let descs = (!schema.entity_descriptions.is_empty()).then_some(&schema.entity_descriptions);
    let mode = if descs.is_some() {
        "descriptions"
    } else {
        "none"
    };
    let toks = transform_schema(
        "entities",
        &chosen,
        crate::processor::E_TOKEN,
        descs,
        &[],
        if mode == "none" {
            "none"
        } else {
            "descriptions"
        },
    );
    schemas.push(toks);
    types.push(TaskType::Entities);
}

fn process_relations(
    schema: &ExtractionSchema,
    schemas: &mut Vec<Vec<String>>,
    types: &mut Vec<TaskType>,
) {
    let mut groups: Vec<(String, Vec<IndexMap<String, String>>)> = Vec::new();
    for item in &schema.relations {
        for (parent, endpoints) in &item.types {
            let mut m = IndexMap::new();
            m.insert("head".into(), endpoints.head.clone());
            m.insert("tail".into(), endpoints.tail.clone());
            if let Some(i) = groups.iter().position(|(p, _)| p == parent) {
                groups[i].1.push(m);
            } else {
                groups.push((parent.clone(), vec![m]));
            }
        }
    }
    for (parent, occurrences) in groups {
        let field_names: Vec<String> = occurrences[0].keys().cloned().collect();
        let toks = transform_schema(&parent, &field_names, "[R]", None, &[], "none");
        schemas.push(toks);
        types.push(TaskType::Relations);
    }
}

fn process_classifications(
    schema: &ExtractionSchema,
    schemas: &mut Vec<Vec<String>>,
    types: &mut Vec<TaskType>,
) -> Result<()> {
    for cls in &schema.classifications {
        let task = cls.task.clone();
        let labels = cls.labels.clone();
        let descs = cls.label_descriptions.as_ref();
        let examples: Vec<(String, String)> = cls
            .examples
            .as_ref()
            .map(|a| {
                a.iter()
                    .filter_map(|ex| match ex {
                        ClassificationExample::List(v) if v.len() >= 2 => {
                            Some((v[0].clone(), v[1].clone()))
                        }
                        ClassificationExample::Obj(o) => Some((o.input.clone(), o.output.clone())),
                        _ => None,
                    })
                    .collect()
            })
            .unwrap_or_default();
        let mode = "both";
        let toks = transform_schema(&task, &labels, "[L]", descs, &examples, mode);
        schemas.push(toks);
        types.push(TaskType::Classifications);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{ExtractionSchema, PreprocessedInput, TaskType, collate_preprocessed};

    fn dummy_pre(seq_len: usize, id_base: u32) -> PreprocessedInput {
        PreprocessedInput {
            input_ids: (0..seq_len as u32).map(|i| id_base + i).collect(),
            text_word_first_positions: vec![0],
            schema_special_indices: vec![],
            schema_tokens_list: vec![],
            task_types: vec![TaskType::Classifications],
            text_tokens: vec![],
            start_mappings: vec![],
            end_mappings: vec![],
            original_text: String::new(),
            len_prefix: 0,
            schema_working: ExtractionSchema::default(),
            schema_original: ExtractionSchema::default(),
        }
    }

    #[test]
    fn collate_empty_returns_none() {
        assert!(collate_preprocessed(&[]).is_none());
    }

    #[test]
    fn collate_pads_to_max_length() {
        let a = dummy_pre(3, 10);
        let b = dummy_pre(5, 100);
        let batch = collate_preprocessed(&[a, b]).unwrap();
        assert_eq!(batch.batch_size, 2);
        assert_eq!(batch.max_seq_len, 5);
        assert_eq!(batch.input_ids.len(), 10);
        assert_eq!(
            batch.attention_mask,
            vec![
                1, 1, 1, 0, 0, //
                1, 1, 1, 1, 1,
            ]
        );
        assert_eq!(&batch.input_ids[0..3], &[10, 11, 12]);
        assert_eq!(&batch.input_ids[5..10], &[100, 101, 102, 103, 104]);
    }
}
