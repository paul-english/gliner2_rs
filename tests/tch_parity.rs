//! Candle vs LibTorch-encoder (`TchExtractor`) NER score parity (ignored by default).
use candle_core::Device;
use candle_nn::VarBuilder;
use candle_transformers::models::debertav2::Config as DebertaConfig;
use gliner2::config::download_model;
use gliner2::{Extractor, ExtractorConfig, SchemaTransformer, TchExtractor};
use tch::Device as TchDevice;

fn max_abs_diff_slices(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "tensor length mismatch");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (*x - *y).abs())
        .fold(0f32, f32::max)
}

#[test]
#[ignore = "downloads ~GB model; needs LibTorch; cargo test -p gliner2 --features tch --test tch_parity -- --ignored"]
fn tch_forward_matches_candle_within_tolerance() {
    let model_id = std::env::var("GLINER2_TEST_MODEL_ID")
        .unwrap_or_else(|_| "fastino/gliner2-base-v1".to_string());

    let files = download_model(&model_id).expect("download_model");
    let device = Device::Cpu;
    let dtype = candle_core::DType::F32;

    let config: ExtractorConfig =
        serde_json::from_str(&std::fs::read_to_string(&files.config).unwrap()).unwrap();
    let enc_json = std::fs::read_to_string(&files.encoder_config).unwrap();

    let mut encoder_candle: DebertaConfig = serde_json::from_str(&enc_json).unwrap();
    let processor = SchemaTransformer::new(files.tokenizer.to_str().unwrap()).unwrap();
    let vocab = processor.tokenizer.get_vocab_size(true);
    encoder_candle.vocab_size = vocab;

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[files.weights.clone()], dtype, &device).unwrap()
    };
    let candle_ext = Extractor::load(config.clone(), encoder_candle, vb).unwrap();

    let mut encoder_tch: DebertaConfig = serde_json::from_str(&enc_json).unwrap();
    encoder_tch.vocab_size = vocab;
    let tch_ext = TchExtractor::load(&files, config, encoder_tch, vocab, TchDevice::Cpu).unwrap();

    let text = "Alice founded Acme Corp in Paris last Tuesday.";
    let entities = ["person", "company", "location", "date"];
    let formatted = processor.format_input_for_ner(text, &entities).unwrap();

    let input_ids = candle_core::Tensor::new(formatted.input_ids.clone(), &device)
        .unwrap()
        .unsqueeze(0)
        .unwrap();
    let attention_mask =
        candle_core::Tensor::ones(input_ids.dims(), candle_core::DType::I64, &device).unwrap();

    let s_c = candle_ext
        .forward(&input_ids, &attention_mask, &formatted)
        .unwrap();
    let s_t = tch_ext
        .forward(&input_ids, &attention_mask, &formatted)
        .unwrap();

    let v_c = s_c.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let v_t = s_t.flatten_all().unwrap().to_vec1::<f32>().unwrap();

    let md = max_abs_diff_slices(&v_c, &v_t);
    assert!(
        md < 5e-3,
        "max abs diff between candle and tch NER scores: {md} (len {})",
        v_c.len()
    );
}
