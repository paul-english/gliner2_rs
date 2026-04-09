use std::env;

fn main() {
    // When we're building with the tch-rs feature
    if env::var_os("CARGO_FEATURE_TCH").is_some() {
        let os = env::var("CARGO_CFG_TARGET_OS").expect("Unable to get TARGET_OS");
        match os.as_str() {
            "linux" | "windows" => {
                // Make sure we configure the binary to link libtorch
                // correctly
                if let Some(lib_path) = std::env::var_os("DEP_TCH_LIBTORCH_LIB") {
                    println!(
                        "cargo:rustc-link-arg=-Wl,-rpath={}",
                        lib_path.to_string_lossy()
                    );
                }
                println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
                println!("cargo:rustc-link-arg=-ltorch");
            }
            _ => {}
        }
    }
}
