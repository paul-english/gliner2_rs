//! Multi-task extraction and `format_results` aligned with `gliner2.inference.engine`.

mod output;

pub use output::{ExtractionOutput, LabelConfidence, TaskValue};

use crate::engine::Gliner2Engine;
use crate::preprocess::{PreprocessedInput, TaskType, collate_preprocessed};
use crate::processor::SchemaTransformer;
use crate::schema::{ClassificationConfig, ExtractionMetadata, ExtractionSchema, StructureField};
use anyhow::{Context, Result};
use indexmap::IndexMap;
use std::collections::{BTreeMap, HashMap, HashSet};

#[derive(Clone, Debug)]
pub struct ExtractOptions {
    pub threshold: f32,
    pub format_results: bool,
    pub include_confidence: bool,
    pub include_spans: bool,
    pub max_len: Option<usize>,
    /// Chunk size for batch extraction APIs (Python default: 8).
    pub batch_size: usize,
}

impl Default for ExtractOptions {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            format_results: true,
            include_confidence: false,
            include_spans: false,
            max_len: None,
            batch_size: 8,
        }
    }
}

/// How schemas and metadata map to each text in [`batch_extract`].
#[derive(Clone, Copy, Debug)]
pub enum BatchSchemaMode<'a> {
    /// One schema and metadata shared by every text in the batch.
    Shared {
        schema: &'a ExtractionSchema,
        meta: &'a ExtractionMetadata,
    },
    /// Per-text schema and metadata (`len` must match `texts.len()`).
    PerSample {
        schemas: &'a [ExtractionSchema],
        metas: &'a [ExtractionMetadata],
    },
}

fn sample_needs_span_rep(pre: &PreprocessedInput) -> bool {
    pre.task_types.iter().any(|t| {
        matches!(
            t,
            TaskType::Entities | TaskType::Relations | TaskType::JsonStructures
        )
    })
}

/// Decode from one sample’s encoder output and optional span tensor (multi-task heads + `format_results`).
pub fn extract_from_preprocessed<E: Gliner2Engine>(
    engine: &E,
    pre: &PreprocessedInput,
    last_hidden_seq: &E::Tensor,
    span_rep: Option<&E::Tensor>,
    meta: &ExtractionMetadata,
    opts: &ExtractOptions,
) -> Result<ExtractionOutput> {
    let span_for_tasks: Option<&E::Tensor> = if sample_needs_span_rep(pre) {
        Some(span_rep.context("extract_from_preprocessed: span_rep required for span tasks")?)
    } else {
        None
    };

    let l_words = pre.text_word_first_positions.len();
    let text_len = pre.start_mappings.len();
    let len_prefix = pre.len_prefix;

    let mut raw: BTreeMap<String, TaskValue> = BTreeMap::new();

    for (task_i, task_type) in pre.task_types.iter().enumerate() {
        let positions = pre
            .schema_special_indices
            .get(task_i)
            .cloned()
            .unwrap_or_default();
        let schema_tokens = pre
            .schema_tokens_list
            .get(task_i)
            .cloned()
            .unwrap_or_default();

        if schema_tokens.len() < 4 || positions.is_empty() {
            continue;
        }

        let embs = engine.stack_schema_token_embeddings(last_hidden_seq, &positions)?;

        match task_type {
            TaskType::Classifications => {
                extract_classification(
                    &mut raw,
                    &schema_tokens,
                    &pre.schema_original,
                    engine,
                    &embs,
                    opts.include_confidence,
                )?;
            }
            TaskType::Entities => {
                extract_span_task(
                    engine,
                    &mut raw,
                    "entities",
                    &schema_tokens,
                    &embs,
                    span_for_tasks.expect("span task without tensor"),
                    task_type,
                    l_words,
                    text_len,
                    len_prefix,
                    &pre.original_text,
                    &pre.text_tokens,
                    &pre.start_mappings,
                    &pre.end_mappings,
                    opts.threshold,
                    meta,
                    &build_cls_fields(&pre.schema_original),
                    opts.include_confidence,
                    opts.include_spans,
                )?;
            }
            TaskType::Relations => {
                let name = schema_prompt_raw(&schema_tokens);
                extract_span_task(
                    engine,
                    &mut raw,
                    &name,
                    &schema_tokens,
                    &embs,
                    span_for_tasks.expect("span task without tensor"),
                    task_type,
                    l_words,
                    text_len,
                    len_prefix,
                    &pre.original_text,
                    &pre.text_tokens,
                    &pre.start_mappings,
                    &pre.end_mappings,
                    opts.threshold,
                    meta,
                    &build_cls_fields(&pre.schema_original),
                    opts.include_confidence,
                    opts.include_spans,
                )?;
            }
            TaskType::JsonStructures => {
                let name = schema_prompt_raw(&schema_tokens);
                extract_span_task(
                    engine,
                    &mut raw,
                    &name,
                    &schema_tokens,
                    &embs,
                    span_for_tasks.expect("span task without tensor"),
                    task_type,
                    l_words,
                    text_len,
                    len_prefix,
                    &pre.original_text,
                    &pre.text_tokens,
                    &pre.start_mappings,
                    &pre.end_mappings,
                    opts.threshold,
                    meta,
                    &build_cls_fields(&pre.schema_original),
                    opts.include_confidence,
                    opts.include_spans,
                )?;
            }
        }
    }

    let fields = if opts.format_results {
        format_results(
            raw,
            opts.include_confidence,
            &meta.relation_order,
            &meta.classification_tasks,
        )
    } else {
        raw
    };
    Ok(ExtractionOutput { fields })
}

/// End-to-end extract: preprocess → encode → heads → optional `format_results`.
pub fn extract_with_schema<E: Gliner2Engine>(
    engine: &E,
    transformer: &SchemaTransformer,
    text: &str,
    schema: &ExtractionSchema,
    meta: &ExtractionMetadata,
    opts: &ExtractOptions,
) -> Result<ExtractionOutput> {
    let pre = transformer.transform_extract(text, schema, opts.max_len)?;
    let (input_ids, attention_mask) = engine.single_sample_inputs(&pre.input_ids)?;

    let hidden = engine.encode_sequence(&input_ids, &attention_mask)?;
    let text_embs = engine.gather_text_word_embeddings(&hidden, &pre.text_word_first_positions)?;
    let span_rep = engine.compute_span_rep(&text_embs)?;

    let last_hidden = engine.batch_row_hidden(&hidden, 0)?;
    let span_opt = sample_needs_span_rep(&pre).then_some(&span_rep);
    extract_from_preprocessed(engine, &pre, &last_hidden, span_opt, meta, opts)
}

/// Batch extract with Python `batch_extract` semantics: padded encoder passes, batched span rep, per-sample decode.
/// Stream extraction results batch-by-batch, calling `on_batch` with
/// `(global_offset, batch_results)` as each inference batch completes.
///
/// This avoids buffering all results in memory and allows the caller to
/// write output incrementally (e.g. JSONL lines or parquet row groups).
pub fn batch_extract_streaming<E, F>(
    engine: &E,
    transformer: &SchemaTransformer,
    texts: &[String],
    mode: BatchSchemaMode<'_>,
    opts: &ExtractOptions,
    mut on_batch: F,
) -> Result<()>
where
    E: Gliner2Engine,
    F: FnMut(usize, Vec<ExtractionOutput>) -> Result<()>,
{
    use rayon::prelude::*;

    if texts.is_empty() {
        return Ok(());
    }

    match mode {
        BatchSchemaMode::PerSample { schemas, metas } => {
            anyhow::ensure!(
                schemas.len() == texts.len(),
                "batch_extract: schemas.len() ({}) != texts.len() ({})",
                schemas.len(),
                texts.len()
            );
            anyhow::ensure!(
                metas.len() == texts.len(),
                "batch_extract: metas.len() ({}) != texts.len() ({})",
                metas.len(),
                texts.len()
            );
        }
        BatchSchemaMode::Shared { .. } => {}
    }

    let bs = opts.batch_size.max(1);
    let mut base = 0usize;

    for chunk in texts.chunks(bs) {
        // -- Parallel preprocess: tokenize + schema transform per record --
        let pre_results: Vec<Result<(crate::preprocess::PreprocessedInput, &ExtractionMetadata)>> =
            chunk
                .par_iter()
                .enumerate()
                .map(|(j, text)| {
                    let global = base + j;
                    let (schema, meta) = match mode {
                        BatchSchemaMode::Shared { schema, meta } => (schema, meta),
                        BatchSchemaMode::PerSample { schemas, metas } => {
                            (&schemas[global], &metas[global])
                        }
                    };
                    let pre = transformer.transform_extract(text.as_str(), schema, opts.max_len)?;
                    Ok((pre, meta))
                })
                .collect();

        let mut pres = Vec::with_capacity(chunk.len());
        let mut metas_chunk: Vec<&ExtractionMetadata> = Vec::with_capacity(chunk.len());
        for r in pre_results {
            let (pre, meta) = r?;
            pres.push(pre);
            metas_chunk.push(meta);
        }

        // -- Batched tensor forward (sequential, single encoder pass) --
        let batch = collate_preprocessed(&pres).expect("non-empty chunk");
        let mask_i64: Vec<i64> = batch.attention_mask.iter().map(|&x| x as i64).collect();
        let (input_ids, attention_mask) = engine.batch_inputs(
            batch.input_ids.clone(),
            mask_i64,
            batch.batch_size,
            batch.max_seq_len,
        )?;

        let hidden = engine.encode_sequence(&input_ids, &attention_mask)?;

        // -- Gather per-sample embeddings + batched span rep (sequential) --
        let mut span_emb_list: Vec<E::Tensor> = Vec::new();
        let mut span_orig_index: Vec<usize> = Vec::new();
        for (i, pre) in batch.samples.iter().enumerate() {
            if sample_needs_span_rep(pre) {
                let tw = engine.gather_text_word_embeddings_batch_idx(
                    &hidden,
                    i,
                    &pre.text_word_first_positions,
                )?;
                span_orig_index.push(i);
                span_emb_list.push(tw);
            }
        }

        let batched_spans = if span_emb_list.is_empty() {
            vec![]
        } else {
            engine.compute_span_rep_batched(&span_emb_list)?
        };

        let mut span_by_sample: Vec<Option<E::Tensor>> =
            (0..batch.batch_size).map(|_| None).collect();
        for (k, &orig_i) in span_orig_index.iter().enumerate() {
            span_by_sample[orig_i] = Some(engine.dup_tensor(&batched_spans[k]));
        }

        // -- Phase A: gather per-sample tensor data (sequential, indexes batched hidden) --
        let sample_data: Vec<(usize, E::Tensor, Option<E::Tensor>)> = (0..batch.batch_size)
            .map(|i| {
                let last_h = engine.batch_row_hidden(&hidden, i)?;
                let span = span_by_sample[i].take();
                Ok((i, last_h, span))
            })
            .collect::<Result<_>>()?;

        // -- Phase B: parallel per-record decode (into_par_iter: each thread owns its tensors) --
        let samples = &batch.samples;
        let decoded: Vec<Result<ExtractionOutput>> = sample_data
            .into_par_iter()
            .map(|(i, last_h, span)| {
                let pre = &samples[i];
                let meta = metas_chunk[i];
                extract_from_preprocessed(engine, pre, &last_h, span.as_ref(), meta, opts)
            })
            .collect();

        let mut batch_results = Vec::with_capacity(decoded.len());
        for v in decoded {
            batch_results.push(v?);
        }

        on_batch(base, batch_results)?;

        base += chunk.len();
    }

    Ok(())
}

/// Collect all extraction results into a `Vec<ExtractionOutput>`.
///
/// This is a convenience wrapper around [`batch_extract_streaming`] for
/// callers that need all results in memory (e.g. for further processing).
pub fn batch_extract<E: Gliner2Engine>(
    engine: &E,
    transformer: &SchemaTransformer,
    texts: &[String],
    mode: BatchSchemaMode<'_>,
    opts: &ExtractOptions,
) -> Result<Vec<ExtractionOutput>> {
    let mut all_out = Vec::with_capacity(texts.len());
    batch_extract_streaming(engine, transformer, texts, mode, opts, |_offset, batch| {
        all_out.extend(batch);
        Ok(())
    })?;
    Ok(all_out)
}

fn schema_prompt_raw(tokens: &[String]) -> String {
    if tokens.len() < 3 {
        return String::new();
    }
    tokens[2]
        .split(" [DESCRIPTION] ")
        .next()
        .unwrap_or(&tokens[2])
        .to_string()
}

fn field_names_from_schema_tokens(tokens: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    for j in 0..tokens.len().saturating_sub(1) {
        if matches!(tokens[j].as_str(), "[E]" | "[C]" | "[R]" | "[L]") {
            out.push(tokens[j + 1].clone());
        }
    }
    out
}

fn build_cls_fields(schema: &ExtractionSchema) -> HashMap<String, Vec<String>> {
    let mut m = HashMap::new();
    for block in &schema.json_structures {
        for (parent, fmap) in &block.parents {
            for (fname, fval) in fmap {
                if let StructureField::Rich(body) = fval {
                    if body.value.is_some() {
                        if let Some(ch) = &body.choices {
                            m.insert(format!("{parent}.{fname}"), ch.clone());
                        }
                    }
                }
            }
        }
    }
    m
}

fn extract_classification<E: Gliner2Engine>(
    results: &mut BTreeMap<String, TaskValue>,
    schema_tokens: &[String],
    schema: &ExtractionSchema,
    engine: &E,
    embs: &E::Tensor,
    include_confidence: bool,
) -> Result<()> {
    let prompt = schema_prompt_raw(schema_tokens);
    let Some(cls) = find_classification_config(schema, &prompt) else {
        return Ok(());
    };
    let n = engine.tensor_dim0(embs)?;
    if n < 2 {
        return Ok(());
    }
    let cls_embeds = engine.tensor_narrow0(embs, 1, n - 1)?;
    let logits = engine.classifier_logits(&cls_embeds)?;
    let logits_v = engine.tensor_logits_1d(&logits)?;
    let labels = cls.labels.clone();
    let multi = cls.multi_label;
    let cls_threshold = cls.cls_threshold;
    let class_act = cls.class_act.as_deref().unwrap_or("auto");

    let schema_name = prompt.clone();

    if multi || class_act == "sigmoid" {
        let use_sigmoid = class_act != "softmax";
        let p: Vec<f32> = if use_sigmoid {
            logits_v.iter().map(|&z| 1.0 / (1.0 + (-z).exp())).collect()
        } else {
            softmax(&logits_v)
        };
        let mut chosen: Vec<(String, f32)> = Vec::new();
        for (j, lab) in labels.iter().enumerate() {
            if p[j] >= cls_threshold {
                chosen.push((lab.clone(), p[j]));
            }
        }
        if chosen.is_empty() {
            let best = argmax_f32(&p);
            chosen.push((labels[best].clone(), p[best]));
        }
        results.insert(
            schema_name,
            if include_confidence {
                TaskValue::LabelConfidenceList(
                    chosen
                        .into_iter()
                        .map(|(l, c)| LabelConfidence {
                            label: l,
                            confidence: c,
                        })
                        .collect(),
                )
            } else {
                TaskValue::StringArray(chosen.into_iter().map(|(l, _)| l).collect())
            },
        );
    } else {
        let p = if class_act == "softmax" {
            softmax(&logits_v)
        } else {
            logits_v.iter().map(|&z| 1.0 / (1.0 + (-z).exp())).collect()
        };
        let best = argmax_f32(&p);
        let label = labels.get(best).cloned().unwrap_or_default();
        let conf = p[best];
        results.insert(
            schema_name,
            if include_confidence {
                TaskValue::LabelConfidence(LabelConfidence {
                    label,
                    confidence: conf,
                })
            } else {
                TaskValue::String(label)
            },
        );
    }
    Ok(())
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&z| (z - m).exp()).collect();
    let s: f32 = exps.iter().sum();
    exps.iter().map(|e| e / s).collect()
}

fn argmax_f32(p: &[f32]) -> usize {
    p.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn find_classification_config<'a>(
    schema: &'a ExtractionSchema,
    prompt: &str,
) -> Option<&'a ClassificationConfig> {
    schema
        .classifications
        .iter()
        .find(|c| prompt.starts_with(&c.task))
}

#[allow(clippy::too_many_arguments)]
fn extract_span_task<E: Gliner2Engine>(
    engine: &E,
    results: &mut BTreeMap<String, TaskValue>,
    schema_name: &str,
    schema_tokens: &[String],
    embs: &E::Tensor,
    span_rep: &E::Tensor,
    task_type: &TaskType,
    l_words: usize,
    text_len: usize,
    len_prefix: usize,
    text: &str,
    text_tokens: &[String],
    start_map: &[usize],
    end_map: &[usize],
    threshold: f32,
    meta: &ExtractionMetadata,
    cls_fields: &HashMap<String, Vec<String>>,
    include_confidence: bool,
    include_spans: bool,
) -> Result<()> {
    let field_names = field_names_from_schema_tokens(schema_tokens);
    if field_names.is_empty() {
        insert_empty_span_result(results, schema_name, task_type);
        return Ok(());
    }

    let p_emb = engine.tensor_index0(embs, 0)?;
    let pred_count = engine.count_predict(&p_emb)?;

    if pred_count == 0 {
        insert_empty_span_result(results, schema_name, task_type);
        return Ok(());
    }

    let n = engine.tensor_dim0(embs)?;
    let field_embs = engine.tensor_narrow0(embs, 1, n - 1)?;
    let span_scores = engine.span_scores_sigmoid(span_rep, &field_embs, pred_count)?;
    let scores_v = engine.tensor_span_scores_to_vec4(&span_scores)?;

    match task_type {
        TaskType::Entities => {
            let ent = extract_entities_inner(
                &field_names,
                &scores_v,
                pred_count,
                l_words,
                text_len,
                len_prefix,
                text,
                text_tokens,
                start_map,
                end_map,
                threshold,
                meta,
                include_confidence,
                include_spans,
            )?;
            results.insert(schema_name.to_string(), ent);
        }
        TaskType::Relations => {
            let rel = extract_relations_inner(
                schema_name,
                &field_names,
                &scores_v,
                pred_count,
                l_words,
                text_len,
                len_prefix,
                text,
                start_map,
                end_map,
                threshold,
                meta,
                include_confidence,
                include_spans,
            )?;
            results.insert(schema_name.to_string(), rel);
        }
        TaskType::JsonStructures => {
            let st = extract_structures_inner(
                schema_name,
                &field_names,
                &scores_v,
                pred_count,
                l_words,
                text_len,
                len_prefix,
                text,
                text_tokens,
                start_map,
                end_map,
                threshold,
                meta,
                cls_fields,
                include_confidence,
                include_spans,
            )?;
            results.insert(schema_name.to_string(), st);
        }
        TaskType::Classifications => {}
    }
    Ok(())
}

fn insert_empty_span_result(
    results: &mut BTreeMap<String, TaskValue>,
    schema_name: &str,
    task_type: &TaskType,
) {
    let v = match task_type {
        TaskType::Entities | TaskType::Relations | TaskType::JsonStructures => {
            TaskValue::ObjectArray(vec![])
        }
        TaskType::Classifications => TaskValue::Null,
    };
    results.insert(schema_name.to_string(), v);
}

#[allow(clippy::too_many_arguments)]
fn extract_entities_inner(
    field_names: &[String],
    span_scores: &[Vec<Vec<Vec<f32>>>],
    _pred_count: usize,
    l_words: usize,
    text_len: usize,
    _len_prefix: usize,
    text: &str,
    _text_tokens: &[String],
    start_map: &[usize],
    end_map: &[usize],
    threshold: f32,
    meta: &ExtractionMetadata,
    include_confidence: bool,
    include_spans: bool,
) -> Result<TaskValue> {
    let b = 0usize;
    let slice_l = l_words.saturating_sub(text_len);
    let mut entity_map: BTreeMap<String, TaskValue> = BTreeMap::new();
    let order: Vec<String> = if meta.entity_order.is_empty() {
        field_names.to_vec()
    } else {
        meta.entity_order.clone()
    };

    for name in order {
        if !field_names.contains(&name) {
            continue;
        }
        let idx = field_names.iter().position(|x| x == &name).unwrap();
        let ent_threshold = meta
            .entity_metadata
            .get(&name)
            .and_then(|m| m.threshold)
            .unwrap_or(threshold);
        let dtype_list = meta
            .entity_metadata
            .get(&name)
            .map(|m| matches!(m.dtype, crate::schema::ValueDtype::List))
            .unwrap_or(true);

        let scores_2d = slice_scores_pk(span_scores, b, idx, slice_l, l_words, text_len);
        let spans = find_spans_from_grid(
            &scores_2d,
            text_len,
            ent_threshold,
            text,
            start_map,
            end_map,
        );
        let formatted = format_spans(&spans, include_confidence, include_spans);
        if dtype_list {
            entity_map.insert(
                name,
                TaskValue::Array(
                    formatted
                        .into_iter()
                        .map(|(t, c, st, en)| {
                            span_to_entity_tv(&t, c, st, en, include_confidence, include_spans)
                        })
                        .collect(),
                ),
            );
        } else {
            let v = if let Some((t, c, st, en)) = spans.first() {
                span_to_entity_tv(t, *c, *st, *en, include_confidence, include_spans)
            } else if include_spans || include_confidence {
                TaskValue::Null
            } else {
                TaskValue::String(String::new())
            };
            entity_map.insert(name, v);
        }
    }

    Ok(TaskValue::ObjectArray(vec![entity_map]))
}

fn span_to_entity_tv(
    text: &str,
    conf: f32,
    start: usize,
    end: usize,
    include_confidence: bool,
    include_spans: bool,
) -> TaskValue {
    match (include_spans, include_confidence) {
        (true, true) => TaskValue::Object(BTreeMap::from([
            ("text".into(), TaskValue::String(text.to_string())),
            ("confidence".into(), TaskValue::F64(conf as f64)),
            ("start".into(), TaskValue::U64(start as u64)),
            ("end".into(), TaskValue::U64(end as u64)),
        ])),
        (true, false) => TaskValue::Object(BTreeMap::from([
            ("text".into(), TaskValue::String(text.to_string())),
            ("start".into(), TaskValue::U64(start as u64)),
            ("end".into(), TaskValue::U64(end as u64)),
        ])),
        (false, true) => TaskValue::Object(BTreeMap::from([
            ("text".into(), TaskValue::String(text.to_string())),
            ("confidence".into(), TaskValue::F64(conf as f64)),
        ])),
        (false, false) => TaskValue::String(text.to_string()),
    }
}

fn slice_scores_pk(
    span_scores: &[Vec<Vec<Vec<f32>>>],
    b: usize,
    p: usize,
    slice_l: usize,
    l_words: usize,
    text_len: usize,
) -> Vec<Vec<f32>> {
    // span_scores[b][p][l][k] with l in 0..l_words
    let empty = vec![vec![0f32; 12]; text_len];
    let Some(plane) = span_scores.get(b) else {
        return empty;
    };
    let Some(row) = plane.get(p) else {
        return empty;
    };
    let max_w = row.first().map(|r| r.len()).unwrap_or(0);
    let mut out = vec![vec![0f32; max_w]; text_len];
    for (ti, l) in (slice_l..l_words).enumerate() {
        if ti >= text_len {
            break;
        }
        if let Some(sc) = row.get(l) {
            for (k, &v) in sc.iter().enumerate() {
                if k < max_w {
                    out[ti][k] = v;
                }
            }
        }
    }
    out
}

fn find_spans_from_grid(
    scores: &[Vec<f32>],
    text_len: usize,
    threshold: f32,
    text: &str,
    start_map: &[usize],
    end_map: &[usize],
) -> Vec<(String, f32, usize, usize)> {
    let max_w = scores.first().map(|r| r.len()).unwrap_or(0);
    let mut spans = Vec::new();
    for (start, row) in scores.iter().enumerate().take(text_len) {
        for (width, &conf) in row.iter().enumerate().take(max_w) {
            if conf < threshold {
                continue;
            }
            let end = start + width + 1;
            if start < text_len && end <= text_len {
                let (cs, ce) = match (start_map.get(start), end_map.get(end.saturating_sub(1))) {
                    (Some(&cs), Some(&ce)) => (cs, ce),
                    _ => continue,
                };
                let text_span = text.get(cs..ce).unwrap_or("").trim();
                if !text_span.is_empty() {
                    spans.push((text_span.to_string(), conf, cs, ce));
                }
            }
        }
    }
    spans
}

fn format_spans(
    spans: &[(String, f32, usize, usize)],
    _include_confidence: bool,
    _include_spans: bool,
) -> Vec<(String, f32, usize, usize)> {
    let mut sorted: Vec<_> = spans.to_vec();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let mut selected = Vec::new();
    for s in sorted {
        let overlap = selected
            .iter()
            .any(|t: &(String, f32, usize, usize)| !(s.3 <= t.2 || s.2 >= t.3));
        if !overlap {
            selected.push(s);
        }
    }
    selected.sort_by_key(|s| (s.2, s.3));
    selected
}

#[allow(clippy::too_many_arguments)]
fn extract_relations_inner(
    rel_name: &str,
    field_names: &[String],
    span_scores: &[Vec<Vec<Vec<f32>>>],
    pred_count: usize,
    l_words: usize,
    text_len: usize,
    _len_prefix: usize,
    text: &str,
    start_map: &[usize],
    end_map: &[usize],
    threshold: f32,
    meta: &ExtractionMetadata,
    include_confidence: bool,
    include_spans: bool,
) -> Result<TaskValue> {
    let slice_l = l_words.saturating_sub(text_len);
    let rel_threshold = meta
        .relation_metadata
        .get(rel_name)
        .and_then(|m| m.threshold)
        .unwrap_or(threshold);
    let ordered: Vec<String> = meta
        .field_orders
        .get(rel_name)
        .cloned()
        .unwrap_or_else(|| field_names.to_vec());

    let mut instances: Vec<TaskValue> = Vec::new();
    for inst in 0..pred_count {
        let mut values: Vec<Option<String>> = Vec::new();
        let mut field_data: Vec<Option<(String, f32, usize, usize)>> = Vec::new();

        for fname in &ordered {
            if !field_names.contains(fname) {
                continue;
            }
            let fidx = field_names.iter().position(|x| x == fname).unwrap();
            let scores_2d = slice_scores_pk(span_scores, inst, fidx, slice_l, l_words, text_len);
            let spans = find_spans_from_grid(
                &scores_2d,
                text_len,
                rel_threshold,
                text,
                start_map,
                end_map,
            );
            if let Some((t, c, cs, ce)) = spans.first() {
                values.push(Some(t.clone()));
                field_data.push(Some((t.clone(), *c, *cs, *ce)));
            } else {
                values.push(None);
                field_data.push(None);
            }
        }

        if values.len() == 2 && values[0].is_some() && values[1].is_some() {
            let fd0 = field_data[0].as_ref().unwrap();
            let fd1 = field_data[1].as_ref().unwrap();
            let tup = if include_spans && include_confidence {
                TaskValue::Object(BTreeMap::from([
                    (
                        "head".into(),
                        span_to_entity_tv(&fd0.0, fd0.1, fd0.2, fd0.3, true, true),
                    ),
                    (
                        "tail".into(),
                        span_to_entity_tv(&fd1.0, fd1.1, fd1.2, fd1.3, true, true),
                    ),
                ]))
            } else if include_spans {
                TaskValue::Object(BTreeMap::from([
                    (
                        "head".into(),
                        span_to_entity_tv(&fd0.0, fd0.1, fd0.2, fd0.3, false, true),
                    ),
                    (
                        "tail".into(),
                        span_to_entity_tv(&fd1.0, fd1.1, fd1.2, fd1.3, false, true),
                    ),
                ]))
            } else if include_confidence {
                TaskValue::Object(BTreeMap::from([
                    (
                        "head".into(),
                        span_to_entity_tv(&fd0.0, fd0.1, fd0.2, fd0.3, true, false),
                    ),
                    (
                        "tail".into(),
                        span_to_entity_tv(&fd1.0, fd1.1, fd1.2, fd1.3, true, false),
                    ),
                ]))
            } else {
                TaskValue::StringArray(vec![values[0].clone().unwrap(), values[1].clone().unwrap()])
            };
            instances.push(tup);
        }
    }
    Ok(TaskValue::Array(instances))
}

fn find_choice_idx(choice: &str, tokens: &[String]) -> isize {
    let cl = choice.to_lowercase();
    for (i, tok) in tokens.iter().enumerate() {
        let tl = tok.to_lowercase();
        if tl == cl || tl.contains(&cl) {
            return i as isize;
        }
    }
    -1
}

#[allow(clippy::too_many_arguments)]
fn extract_structures_inner(
    struct_name: &str,
    field_names: &[String],
    span_scores: &[Vec<Vec<Vec<f32>>>],
    pred_count: usize,
    l_words: usize,
    text_len: usize,
    _len_prefix: usize,
    text: &str,
    text_tokens: &[String],
    start_map: &[usize],
    end_map: &[usize],
    threshold: f32,
    meta: &ExtractionMetadata,
    cls_fields: &HashMap<String, Vec<String>>,
    include_confidence: bool,
    include_spans: bool,
) -> Result<TaskValue> {
    let slice_l = l_words.saturating_sub(text_len);
    let ordered: Vec<String> = meta
        .field_orders
        .get(struct_name)
        .cloned()
        .unwrap_or_else(|| field_names.to_vec());

    let mut instances: Vec<BTreeMap<String, TaskValue>> = Vec::new();
    for inst in 0..pred_count {
        let mut instance: BTreeMap<String, TaskValue> = BTreeMap::new();
        for fname in &ordered {
            if !field_names.contains(fname) {
                continue;
            }
            let fidx = field_names.iter().position(|x| x == fname).unwrap();
            let field_key = format!("{struct_name}.{fname}");
            let fmeta = meta.field_metadata.get(&field_key);
            let field_threshold = fmeta.and_then(|m| m.threshold).unwrap_or(threshold);
            let dtype_list = fmeta
                .map(|m| matches!(m.dtype, crate::schema::ValueDtype::List))
                .unwrap_or(true);

            if let Some(choices) = cls_fields.get(&field_key) {
                let plane = &span_scores[inst][fidx];
                let prefix_len = l_words.saturating_sub(text_len);
                let mut selected: Vec<TaskValue> = Vec::new();
                let mut seen = HashSet::new();
                if dtype_list {
                    for choice in choices {
                        if !seen.insert(choice.clone()) {
                            continue;
                        }
                        let idx = find_choice_idx(
                            choice,
                            &text_tokens[..prefix_len.min(text_tokens.len())],
                        );
                        if idx >= 0 && (idx as usize) < plane.len() {
                            let row = &plane[idx as usize];
                            let score = row.first().copied().unwrap_or(0.);
                            if score >= field_threshold {
                                if include_confidence {
                                    selected.push(TaskValue::Object(BTreeMap::from([
                                        ("text".into(), TaskValue::String(choice.clone())),
                                        ("confidence".into(), TaskValue::F64(score as f64)),
                                    ])));
                                } else {
                                    selected.push(TaskValue::String(choice.clone()));
                                }
                            }
                        }
                    }
                    instance.insert(fname.clone(), TaskValue::Array(selected));
                } else {
                    let mut best: Option<&str> = None;
                    let mut best_score = -1f32;
                    for choice in choices {
                        let idx = find_choice_idx(
                            choice,
                            &text_tokens[..prefix_len.min(text_tokens.len())],
                        );
                        if idx >= 0 && (idx as usize) < plane.len() {
                            let row = &plane[idx as usize];
                            let score = row.first().copied().unwrap_or(0.);
                            if score > best_score {
                                best_score = score;
                                best = Some(choice.as_str());
                            }
                        }
                    }
                    let v = if let Some(b) = best {
                        if best_score >= field_threshold {
                            if include_confidence {
                                TaskValue::Object(BTreeMap::from([
                                    ("text".into(), TaskValue::String(b.to_string())),
                                    ("confidence".into(), TaskValue::F64(best_score as f64)),
                                ]))
                            } else {
                                TaskValue::String(b.to_string())
                            }
                        } else {
                            TaskValue::Null
                        }
                    } else {
                        TaskValue::Null
                    };
                    instance.insert(fname.clone(), v);
                }
            } else {
                let scores_2d =
                    slice_scores_pk(span_scores, inst, fidx, slice_l, l_words, text_len);
                let mut spans = find_spans_from_grid(
                    &scores_2d,
                    text_len,
                    field_threshold,
                    text,
                    start_map,
                    end_map,
                );
                if let Some(vm) = fmeta {
                    if !vm.validators.is_empty() {
                        spans.retain(|(t, _, _, _)| vm.validators.iter().all(|v| v.validate(t)));
                    }
                }
                let formatted = format_spans(&spans, include_confidence, include_spans);
                if dtype_list {
                    let arr: Vec<TaskValue> = formatted
                        .into_iter()
                        .map(|(t, c, st, en)| {
                            span_to_entity_tv(&t, c, st, en, include_confidence, include_spans)
                        })
                        .collect();
                    instance.insert(fname.clone(), TaskValue::Array(arr));
                } else {
                    let v = if let Some((t, c, st, en)) = formatted.first() {
                        span_to_entity_tv(t, *c, *st, *en, include_confidence, include_spans)
                    } else {
                        TaskValue::Null
                    };
                    instance.insert(fname.clone(), v);
                }
            }
        }
        let has_content = instance.values().any(|v| match v {
            TaskValue::Null => false,
            TaskValue::Array(a) => !a.is_empty(),
            TaskValue::ObjectArray(a) => !a.is_empty(),
            _ => true,
        });
        if has_content {
            instances.push(instance);
        }
    }
    Ok(TaskValue::ObjectArray(instances))
}

fn format_results(
    results: BTreeMap<String, TaskValue>,
    include_confidence: bool,
    requested_relations: &[String],
    classification_tasks: &[String],
) -> BTreeMap<String, TaskValue> {
    let mut formatted: BTreeMap<String, TaskValue> = BTreeMap::new();
    let mut relations: BTreeMap<String, TaskValue> = BTreeMap::new();
    let cls_set: HashSet<&str> = classification_tasks.iter().map(|s| s.as_str()).collect();

    fn is_relation_shape(v: &TaskValue) -> bool {
        match v {
            TaskValue::Array(a) if !a.is_empty() => match a.first() {
                Some(TaskValue::StringArray(inner)) => inner.len() == 2,
                Some(TaskValue::Object(o)) => o.contains_key("head") && o.contains_key("tail"),
                _ => false,
            },
            _ => false,
        }
    }

    for (key, value) in results.into_iter() {
        let is_classification = cls_set.contains(key.as_str());
        let is_relation = if !is_classification {
            if requested_relations.iter().any(|r| r == &key) {
                true
            } else {
                is_relation_shape(&value)
            }
        } else {
            false
        };

        if is_classification {
            let out = match value {
                TaskValue::LabelConfidenceList(list) => {
                    if include_confidence {
                        TaskValue::LabelConfidenceList(list)
                    } else {
                        TaskValue::StringArray(list.into_iter().map(|x| x.label).collect())
                    }
                }
                TaskValue::StringArray(a) => TaskValue::StringArray(a),
                TaskValue::Array(a) => {
                    if include_confidence {
                        TaskValue::Array(a)
                    } else {
                        let labs: Vec<String> = a
                            .into_iter()
                            .filter_map(|x| match x {
                                TaskValue::String(s) => Some(s),
                                TaskValue::LabelConfidence(l) => Some(l.label),
                                TaskValue::Object(mut o) => {
                                    o.remove("label").and_then(|v| match v {
                                        TaskValue::String(s) => Some(s),
                                        _ => None,
                                    })
                                }
                                _ => None,
                            })
                            .collect();
                        TaskValue::StringArray(labs)
                    }
                }
                TaskValue::LabelConfidence(l) => {
                    if include_confidence {
                        TaskValue::LabelConfidence(l)
                    } else {
                        TaskValue::String(l.label)
                    }
                }
                TaskValue::Object(o) if include_confidence => TaskValue::Object(o.clone()),
                TaskValue::Object(o) => {
                    if let Some(l) = o.get("label").cloned() {
                        l
                    } else {
                        TaskValue::Object(o.clone())
                    }
                }
                other => other,
            };
            formatted.insert(key, out);
        } else if is_relation {
            if let TaskValue::Array(a) = value {
                relations.insert(key, TaskValue::Array(a));
            } else {
                relations.insert(key, TaskValue::Array(vec![]));
            }
        } else if let TaskValue::ObjectArray(a) = value {
            if a.is_empty() {
                if key == "entities" {
                    formatted.insert(key, TaskValue::Object(BTreeMap::new()));
                } else {
                    formatted.insert(key, TaskValue::ObjectArray(a));
                }
            } else if let Some(ent) = a.first() {
                if key == "entities" {
                    formatted.insert(key, format_entity_dict(ent, include_confidence));
                } else {
                    let mapped: Vec<BTreeMap<String, TaskValue>> = a
                        .iter()
                        .map(|o| format_struct_obj(o, include_confidence))
                        .collect();
                    formatted.insert(key, TaskValue::ObjectArray(mapped));
                }
            } else {
                formatted.insert(key, TaskValue::ObjectArray(a));
            }
        } else if let TaskValue::Array(a) = value {
            if a.is_empty() && key == "entities" {
                formatted.insert(key, TaskValue::Object(BTreeMap::new()));
            } else {
                formatted.insert(key, TaskValue::Array(a));
            }
        } else {
            formatted.insert(key, value);
        }
    }

    for rel in requested_relations {
        if !relations.contains_key(rel.as_str()) {
            relations.insert(rel.clone(), TaskValue::Array(vec![]));
        }
    }

    if !relations.is_empty() {
        formatted.insert("relation_extraction".into(), TaskValue::Object(relations));
    }

    formatted
}

fn format_entity_dict(ent: &BTreeMap<String, TaskValue>, include_confidence: bool) -> TaskValue {
    let mut out: BTreeMap<String, TaskValue> = BTreeMap::new();
    for (k, v) in ent {
        let cleaned = match v {
            TaskValue::Array(items) => {
                let mut unique: Vec<TaskValue> = Vec::new();
                let mut seen = HashSet::new();
                for it in items {
                    let text_key = match it {
                        TaskValue::String(s) => Some((
                            s.to_lowercase(),
                            format_string_list_item(s, include_confidence),
                        )),
                        TaskValue::Object(o) => o.get("text").and_then(|t| match t {
                            TaskValue::String(ts) => {
                                let lk = ts.to_lowercase();
                                Some((lk.clone(), it.clone()))
                            }
                            _ => None,
                        }),
                        _ => None,
                    };
                    if let Some((lk, val)) = text_key
                        && seen.insert(lk)
                    {
                        unique.push(val);
                    }
                }
                TaskValue::Array(unique)
            }
            _ => v.clone(),
        };
        out.insert(k.clone(), cleaned);
    }
    TaskValue::Object(out)
}

fn format_string_list_item(s: &str, include_confidence: bool) -> TaskValue {
    if include_confidence {
        TaskValue::Object(BTreeMap::from([
            ("text".into(), TaskValue::String(s.to_string())),
            ("confidence".into(), TaskValue::F64(1.0)),
        ]))
    } else {
        TaskValue::String(s.to_string())
    }
}

fn format_struct_obj(
    o: &BTreeMap<String, TaskValue>,
    include_confidence: bool,
) -> BTreeMap<String, TaskValue> {
    let mut m = BTreeMap::new();
    for (k, v) in o {
        m.insert(k.clone(), format_struct_value(v, include_confidence));
    }
    m
}

fn format_struct_value(v: &TaskValue, include_confidence: bool) -> TaskValue {
    match v {
        TaskValue::Array(a) => TaskValue::Array(
            a.iter()
                .map(|it| format_struct_value(it, include_confidence))
                .collect(),
        ),
        TaskValue::Object(o) => {
            if o.contains_key("text") && o.contains_key("confidence") && !include_confidence {
                o.get("text").cloned().unwrap_or(TaskValue::Null)
            } else {
                TaskValue::Object(o.clone())
            }
        }
        _ => v.clone(),
    }
}

macro_rules! impl_gliner2_api {
    ($t:ty) => {
        impl $t {
            /// High-level API matching Python `Extractor.extract` / `GLiNER2.extract`.
            pub fn extract(
                &self,
                transformer: &SchemaTransformer,
                text: &str,
                schema: &crate::schema::ExtractionSchema,
                meta: &ExtractionMetadata,
                opts: &ExtractOptions,
            ) -> Result<ExtractionOutput> {
                extract_with_schema(self, transformer, text, schema, meta, opts)
            }

            /// [`batch_extract`] with a single shared schema (Python `batch_extract` with one schema dict).
            pub fn batch_extract(
                &self,
                transformer: &SchemaTransformer,
                texts: &[String],
                schema: &crate::schema::ExtractionSchema,
                meta: &ExtractionMetadata,
                opts: &ExtractOptions,
            ) -> Result<Vec<ExtractionOutput>> {
                batch_extract(
                    self,
                    transformer,
                    texts,
                    BatchSchemaMode::Shared { schema, meta },
                    opts,
                )
            }

            /// [`batch_extract`] with per-text schemas and metadata.
            pub fn batch_extract_per_sample(
                &self,
                transformer: &SchemaTransformer,
                texts: &[String],
                schemas: &[crate::schema::ExtractionSchema],
                metas: &[ExtractionMetadata],
                opts: &ExtractOptions,
            ) -> Result<Vec<ExtractionOutput>> {
                batch_extract(
                    self,
                    transformer,
                    texts,
                    BatchSchemaMode::PerSample { schemas, metas },
                    opts,
                )
            }

            pub fn batch_extract_entities(
                &self,
                transformer: &SchemaTransformer,
                texts: &[String],
                entity_types: &[String],
                opts: &ExtractOptions,
            ) -> Result<Vec<ExtractionOutput>> {
                let mut s = crate::schema::Schema::new();
                s.entities(crate::schema::EntityTypesInput::Many(entity_types.to_vec()));
                let (schema_val, meta) = s.build();
                self.batch_extract(transformer, texts, &schema_val, &meta, opts)
            }

            pub fn batch_classify_text(
                &self,
                transformer: &SchemaTransformer,
                texts: &[String],
                task: &str,
                labels: crate::schema::ClassificationLabelsInput,
                opts: &ExtractOptions,
            ) -> Result<Vec<ExtractionOutput>> {
                let mut s = crate::schema::Schema::new();
                s.classification_simple(task, labels);
                let (schema_val, meta) = s.build();
                self.batch_extract(transformer, texts, &schema_val, &meta, opts)
            }

            pub fn batch_extract_relations(
                &self,
                transformer: &SchemaTransformer,
                texts: &[String],
                relation_types: crate::schema::RelationTypesInput,
                opts: &ExtractOptions,
            ) -> Result<Vec<ExtractionOutput>> {
                let mut s = crate::schema::Schema::new();
                s.relations(relation_types);
                let (schema_val, meta) = s.build();
                self.batch_extract(transformer, texts, &schema_val, &meta, opts)
            }

            pub fn batch_extract_json(
                &self,
                transformer: &SchemaTransformer,
                texts: &[String],
                structures: &IndexMap<String, Vec<crate::schema::FieldSpecSource>>,
                opts: &ExtractOptions,
            ) -> Result<Vec<ExtractionOutput>> {
                let mut s = crate::schema::Schema::new();
                s.extract_json_structures(structures)?;
                let (schema_val, meta) = s.build();
                self.batch_extract(transformer, texts, &schema_val, &meta, opts)
            }

            pub fn extract_entities(
                &self,
                transformer: &SchemaTransformer,
                text: &str,
                entity_types: &[String],
                opts: &ExtractOptions,
            ) -> Result<ExtractionOutput> {
                let mut s = crate::schema::Schema::new();
                s.entities(crate::schema::EntityTypesInput::Many(entity_types.to_vec()));
                let (schema_val, meta) = s.build();
                self.extract(transformer, text, &schema_val, &meta, opts)
            }

            pub fn classify_text(
                &self,
                transformer: &SchemaTransformer,
                text: &str,
                task: &str,
                labels: crate::schema::ClassificationLabelsInput,
                opts: &ExtractOptions,
            ) -> Result<ExtractionOutput> {
                let mut s = crate::schema::Schema::new();
                s.classification_simple(task, labels);
                let (schema_val, meta) = s.build();
                self.extract(transformer, text, &schema_val, &meta, opts)
            }

            pub fn extract_relations(
                &self,
                transformer: &SchemaTransformer,
                text: &str,
                relation_types: crate::schema::RelationTypesInput,
                opts: &ExtractOptions,
            ) -> Result<ExtractionOutput> {
                let mut s = crate::schema::Schema::new();
                s.relations(relation_types);
                let (schema_val, meta) = s.build();
                self.extract(transformer, text, &schema_val, &meta, opts)
            }

            /// Structured extraction using string or object field specs (Python `GLiNER2.extract_json`).
            pub fn extract_json(
                &self,
                transformer: &SchemaTransformer,
                text: &str,
                structures: &IndexMap<String, Vec<crate::schema::FieldSpecSource>>,
                opts: &ExtractOptions,
            ) -> Result<ExtractionOutput> {
                let mut s = crate::schema::Schema::new();
                s.extract_json_structures(structures)?;
                let (schema_val, meta) = s.build();
                self.extract(transformer, text, &schema_val, &meta, opts)
            }
        }
    };
}

#[cfg(feature = "candle")]
impl_gliner2_api!(crate::CandleExtractor);

#[cfg(feature = "tch")]
impl_gliner2_api!(crate::TchExtractor);
