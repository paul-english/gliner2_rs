use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use gliner2::{Extractor, SchemaTransformer, ExtractorConfig};
use gliner2::config::download_model;
use candle_transformers::models::debertav2::Config as DebertaConfig;
use std::fs;

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let repo_id = "fastino/gliner2-base-v1";
    println!("Downloading model from {}...", repo_id);
    let files = download_model(repo_id)?;
    
    let device = Device::Cpu; // PoC on CPU
    let dtype = candle_core::DType::F32;
    
    // 1. Load Configs
    let config: ExtractorConfig = serde_json::from_str(&fs::read_to_string(&files.config)?)?;
    let mut encoder_config: DebertaConfig = serde_json::from_str(&fs::read_to_string(&files.encoder_config)?)?;
    
    // 2. Load Tokenizer & Processor
    let processor = SchemaTransformer::new(files.tokenizer.to_str().unwrap())?;
    
    // Update vocab_size if needed
    encoder_config.vocab_size = processor.tokenizer.get_vocab_size(true);
    
    // 3. Load Weights
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[files.weights], dtype, &device)?
    };
    
    // 4. Create Extractor
    let extractor = Extractor::load(config, encoder_config, vb)?;
    
    // 5. Inference
    let text = "Steve Jobs founded Apple in Cupertino.";
    let entities = vec!["person", "company", "location"];
    
    println!("Extracting entities from: \"{}\"", text);
    println!("Target entities: {:?}", entities);
    
    let formatted = processor.format_input_for_ner(text, &entities)?;
    
    let input_ids = Tensor::new(formatted.input_ids.clone(), &device)?.unsqueeze(0)?;
    let attention_mask = Tensor::ones(input_ids.dims(), candle_core::DType::I64, &device)?;
    
    let scores = extractor.forward(&input_ids, &attention_mask, &formatted)?;
    
    let results = gliner2::decode::find_spans(
        &scores,
        0.5,
        &entities,
        text,
        &formatted.start_offsets,
        &formatted.end_offsets,
    )?;
    
    println!("\nResults:");
    for entity in results {
        println!("{}: {} ({:.4})", entity.label, entity.text, entity.confidence);
    }

    Ok(())
}
