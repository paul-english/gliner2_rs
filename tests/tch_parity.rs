//! Candle vs LibTorch (`TchExtractor`) NER score parity (ignored by default).
use gliner2::config::download_model;
use gliner2::engine::Gliner2Engine;
use gliner2::{CandleExtractor, ExtractorConfig, SchemaTransformer, TchExtractor};
use tch::Device as TchDevice;

fn max_abs_diff_slices(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "tensor length mismatch");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (*x - *y).abs())
        .fold(0f32, f32::max)
}

fn tch_scores_to_vec(s: &tch::Tensor) -> Vec<f32> {
    let n = s.numel();
    let mut v = vec![0f32; n];
    let dim = s.dim() as i64;
    let flat = if dim <= 1 {
        s.contiguous().view([-1])
    } else {
        s.contiguous().flatten(0, dim - 1)
    };
    flat.copy_data(&mut v, n);
    v
}

#[test]
#[ignore = "downloads ~GB model; needs LibTorch; cargo test -p gliner2 --features \"candle tch\" --test tch_parity -- --ignored"]
fn tch_forward_matches_candle_within_tolerance() {
    let model_id = std::env::var("GLINER2_TEST_MODEL_ID")
        .unwrap_or_else(|_| "fastino/gliner2-base-v1".to_string());

    let files = download_model(&model_id).expect("download_model");

    let config: ExtractorConfig =
        serde_json::from_str(&std::fs::read_to_string(&files.config).unwrap()).unwrap();

    let processor = SchemaTransformer::new(files.tokenizer.to_str().unwrap()).unwrap();
    let vocab = processor.tokenizer.get_vocab_size(true);

    let candle_ext = CandleExtractor::load_cpu(&files, config.clone(), vocab).unwrap();
    let tch_ext = TchExtractor::load(&files, config, vocab, TchDevice::Cpu).unwrap();

    let text = "Alice founded Acme Corp in Paris last Tuesday.";
    let entities = ["person", "company", "location", "date"];
    let formatted = processor.format_input_for_ner(text, &entities).unwrap();

    let (input_ids, attention_mask) = candle_ext
        .single_sample_inputs(&formatted.input_ids)
        .unwrap();
    let s_c = candle_ext
        .forward(&input_ids, &attention_mask, &formatted)
        .unwrap();

    let (ids_t, mask_t) = tch_ext.single_sample_inputs(&formatted.input_ids).unwrap();
    let s_t = tch_ext.forward(&ids_t, &mask_t, &formatted).unwrap();

    let v_c = s_c.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let v_t = tch_scores_to_vec(&s_t);

    let md = max_abs_diff_slices(&v_c, &v_t);
    assert!(
        md < 5e-3,
        "max abs diff between candle and tch NER scores: {md} (len {})",
        v_c.len()
    );
}
