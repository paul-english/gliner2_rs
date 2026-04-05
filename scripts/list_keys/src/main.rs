use candle_core::safetensors::load;
use candle_core::{Device, Result};
use std::env;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: cargo run -p list-keys -- <safetensors_file>");
        return Ok(());
    }
    let tensors = load(&args[1], &Device::Cpu)?;
    let mut keys: Vec<_> = tensors.keys().collect();
    keys.sort();
    for key in keys {
        println!("{}", key);
    }
    Ok(())
}
