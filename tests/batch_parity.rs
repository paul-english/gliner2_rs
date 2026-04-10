//! Batch vs sequential extraction parity (downloads model on first run).
use gliner2::config::download_model;
use gliner2::schema::{ExtractionSchema, infer_metadata_from_schema};
use gliner2::{
    BatchSchemaMode, CandleExtractor, ExtractOptions, ExtractorConfig, SchemaTransformer,
    batch_extract, extract_with_schema,
};
use serde_json::json;

#[test]
#[ignore = "downloads ~GB model from Hugging Face; run: cargo test -p gliner2 --test batch_parity -- --ignored"]
fn batch_extract_matches_sequential_shared_schema() {
    let model_id = std::env::var("GLINER2_TEST_MODEL_ID")
        .unwrap_or_else(|_| "fastino/gliner2-base-v1".to_string());

    let files = download_model(&model_id).expect("download_model");

    let config: ExtractorConfig =
        serde_json::from_str(&std::fs::read_to_string(&files.config).unwrap()).unwrap();
    let processor = SchemaTransformer::new(files.tokenizer.to_str().unwrap()).unwrap();
    let vocab = processor.tokenizer.get_vocab_size(true);

    let extractor = CandleExtractor::load_cpu(&files, config, vocab).unwrap();

    let schema: ExtractionSchema = serde_json::from_value(json!({
        "entities": { "person": "", "company": "" }
    }))
    .unwrap();
    let meta = infer_metadata_from_schema(&schema);

    let texts: Vec<String> = vec![
        "Alice founded Acme in Paris.".into(),
        "Bob works for Contoso.".into(),
    ];

    let opts = ExtractOptions {
        batch_size: 2,
        ..Default::default()
    };

    let mut sequential = Vec::new();
    for t in &texts {
        let v = extract_with_schema(&extractor, &processor, t, &schema, &meta, &opts).unwrap();
        sequential.push(v);
    }

    let batched = batch_extract(
        &extractor,
        &processor,
        &texts,
        BatchSchemaMode::Shared {
            schema: &schema,
            meta: &meta,
        },
        &opts,
    )
    .unwrap();

    assert_eq!(batched.len(), sequential.len());
    for (a, b) in batched.iter().zip(sequential.iter()) {
        assert_eq!(a, b);
    }
}

#[test]
#[ignore = "downloads ~GB model from Hugging Face; run: cargo test -p gliner2 --test batch_parity -- --ignored"]
fn batch_extract_matches_sequential_per_sample_schema() {
    let model_id = std::env::var("GLINER2_TEST_MODEL_ID")
        .unwrap_or_else(|_| "fastino/gliner2-base-v1".to_string());

    let files = download_model(&model_id).expect("download_model");

    let config: ExtractorConfig =
        serde_json::from_str(&std::fs::read_to_string(&files.config).unwrap()).unwrap();
    let processor = SchemaTransformer::new(files.tokenizer.to_str().unwrap()).unwrap();
    let vocab = processor.tokenizer.get_vocab_size(true);

    let extractor = CandleExtractor::load_cpu(&files, config, vocab).unwrap();

    let s0: ExtractionSchema =
        serde_json::from_value(json!({ "entities": { "person": "", "company": "" } })).unwrap();
    let s1: ExtractionSchema =
        serde_json::from_value(json!({ "entities": { "location": "" } })).unwrap();
    let meta0 = infer_metadata_from_schema(&s0);
    let meta1 = infer_metadata_from_schema(&s1);
    let schemas = vec![s0, s1];
    let metas = vec![meta0, meta1];

    let texts: Vec<String> = vec!["Alice works at Acme.".into(), "They met in Berlin.".into()];

    let opts = ExtractOptions {
        batch_size: 2,
        ..Default::default()
    };

    let seq0 = extract_with_schema(
        &extractor,
        &processor,
        &texts[0],
        &schemas[0],
        &metas[0],
        &opts,
    )
    .unwrap();
    let seq1 = extract_with_schema(
        &extractor,
        &processor,
        &texts[1],
        &schemas[1],
        &metas[1],
        &opts,
    )
    .unwrap();

    let batched = batch_extract(
        &extractor,
        &processor,
        &texts,
        BatchSchemaMode::PerSample {
            schemas: &schemas,
            metas: &metas,
        },
        &opts,
    )
    .unwrap();

    assert_eq!(batched.len(), 2);
    assert_eq!(&batched[0], &seq0);
    assert_eq!(&batched[1], &seq1);
}
