use safetensors::SafeTensors;
use std::env;
use std::fs;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: cargo run -p list-keys -- <safetensors_file>");
        return Ok(());
    }
    let bytes = fs::read(&args[1])?;
    let st = SafeTensors::deserialize(&bytes)?;
    let mut keys: Vec<String> = st.names().iter().map(|s| (*s).to_string()).collect();
    keys.sort();
    for key in keys {
        println!("{key}");
    }
    Ok(())
}
