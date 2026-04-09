use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

/// PyTorch version corresponding to tch 0.24.0.
const PYTORCH_VERSION: &str = "2.11.0";

/// Known libtorch compute variants and their download URL slugs.
const VARIANTS: &[(&str, &str, &str)] = &[
    ("cpu", "CPU", "cpu"),
    ("cu126", "CUDA 12.6", "cu126"),
    ("cu128", "CUDA 12.8", "cu128"),
    ("cu130", "CUDA 13.0", "cu130"),
    ("rocm72", "ROCm 7.2", "rocm7.2"),
];

#[derive(Debug, Serialize, Deserialize)]
pub struct AppConfig {
    pub backend: BackendConfig,
    #[serde(default)]
    pub tch: Option<TchConfig>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BackendConfig {
    pub default: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TchConfig {
    pub variant: String,
    pub lib_path: PathBuf,
    pub libtorch_version: String,
}

/// Return the config directory: `$XDG_CONFIG_HOME/gliner2` or `~/.config/gliner2`.
pub fn config_dir() -> Result<PathBuf> {
    let base = dirs::config_dir().context("Could not determine config directory")?;
    Ok(base.join("gliner2"))
}

/// Return the path to `config.toml`.
pub fn config_path() -> Result<PathBuf> {
    Ok(config_dir()?.join("config.toml"))
}

/// Load config from disk, or return None if it doesn't exist.
pub fn load_config() -> Result<Option<AppConfig>> {
    let path = config_path()?;
    if !path.exists() {
        return Ok(None);
    }
    let contents =
        fs::read_to_string(&path).with_context(|| format!("Failed to read {}", path.display()))?;
    let cfg: AppConfig =
        toml::from_str(&contents).with_context(|| format!("Failed to parse {}", path.display()))?;
    Ok(Some(cfg))
}

/// Save config to disk.
pub fn save_config(cfg: &AppConfig) -> Result<()> {
    let path = config_path()?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let contents = toml::to_string_pretty(cfg)?;
    fs::write(&path, contents)?;
    Ok(())
}

/// Return the directory where libtorch is extracted.
pub fn libtorch_dir() -> Result<PathBuf> {
    Ok(config_dir()?.join("lib").join("libtorch"))
}

/// Build the download URL for a libtorch variant.
fn libtorch_url(variant_slug: &str) -> String {
    if cfg!(target_os = "macos") && cfg!(target_arch = "aarch64") {
        format!(
            // https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.11.0.zip
            "https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-{PYTORCH_VERSION}.zip"
        )
    } else if cfg!(target_os = "macos") {
        format!(
            "https://download.pytorch.org/libtorch/cpu/libtorch-macos-x86_64-{PYTORCH_VERSION}.zip"
        )
    } else {
        // Linux x86_64
        format!(
            "https://download.pytorch.org/libtorch/{variant_slug}/libtorch-shared-with-deps-{PYTORCH_VERSION}%2B{variant_slug}.zip"
        )
    }
    // Windows?
}

/// Resolve a variant name (e.g. "cpu", "cu128") to its URL slug.
fn resolve_variant_slug(variant: &str) -> Result<&'static str> {
    for &(name, _, slug) in VARIANTS {
        if name == variant {
            return Ok(slug);
        }
    }
    bail!(
        "Unknown variant {:?}. Known: {}",
        variant,
        VARIANTS
            .iter()
            .map(|(n, _, _)| *n)
            .collect::<Vec<_>>()
            .join(", ")
    );
}

/// Run the interactive setup wizard.
pub fn run_setup(
    non_interactive: bool,
    backend_arg: Option<&str>,
    variant_arg: Option<&str>,
) -> Result<()> {
    eprintln!("gliner2 setup — configure inference backend\n");

    // Step 1: choose backend
    let backend = if let Some(b) = backend_arg {
        b.to_string()
    } else if non_interactive {
        "candle".to_string()
    } else {
        prompt_backend()?
    };

    if backend == "candle" {
        let cfg = AppConfig {
            backend: BackendConfig {
                default: "candle".into(),
            },
            tch: None,
        };
        save_config(&cfg)?;
        eprintln!("Configuration saved. Using candle backend (pure Rust, no external deps).");
        return Ok(());
    }

    if backend != "tch" {
        bail!("Unknown backend {:?}. Use \"candle\" or \"tch\".", backend);
    }

    // Step 2: choose variant
    let variant = if let Some(v) = variant_arg {
        v.to_string()
    } else if non_interactive {
        "cpu".to_string()
    } else {
        prompt_variant()?
    };

    let slug = resolve_variant_slug(&variant)?;
    let url = libtorch_url(slug);
    let dest = libtorch_dir()?;
    let lib_path = dest.join("lib");

    // Step 3: check if already downloaded
    if lib_path.exists() && lib_path.join("libtorch_cpu.so").exists() {
        eprintln!("LibTorch already present at {}", lib_path.display());
        if !non_interactive {
            eprint!("Re-download? [y/N] ");
            std::io::stderr().flush()?;
            let mut answer = String::new();
            std::io::stdin().read_line(&mut answer)?;
            if !answer.trim().eq_ignore_ascii_case("y") {
                let cfg = make_tch_config(&variant, &lib_path);
                save_config(&cfg)?;
                eprintln!("Configuration saved.");
                return Ok(());
            }
        }
    }

    // Step 4: download
    eprintln!("Downloading LibTorch ({variant}) from:\n  {url}\n");
    let zip_data = download_with_progress(&url)?;

    // Step 5: extract
    eprintln!("Extracting to {} ...", dest.display());
    if dest.exists() {
        fs::remove_dir_all(&dest)?;
    }
    // The zip extracts a top-level `libtorch/` directory. We extract into the parent.
    let extract_to = dest
        .parent()
        .context("libtorch dest has no parent")?
        .to_path_buf();
    fs::create_dir_all(&extract_to)?;
    extract_zip(&zip_data, &extract_to)?;

    // Step 6: verify
    if !lib_path.join("libtorch_cpu.so").exists()
        && !lib_path.join("libtorch_cpu.dylib").exists()
        && !lib_path.join("torch_cpu.dll").exists()
    {
        bail!(
            "Extraction succeeded but libtorch_cpu library not found in {}. \
             The archive structure may have changed.",
            lib_path.display()
        );
    }
    eprintln!("LibTorch extracted successfully.");

    // Step 7: save config
    let cfg = make_tch_config(&variant, &lib_path);
    save_config(&cfg)?;
    eprintln!("\nConfiguration saved to {}", config_path()?.display());

    eprintln!(
        "\nSetup complete. Use `gliner2 --backend tch <command>` or set the default in config."
    );
    Ok(())
}

fn make_tch_config(variant: &str, lib_path: &Path) -> AppConfig {
    AppConfig {
        backend: BackendConfig {
            default: "tch".into(),
        },
        tch: Some(TchConfig {
            variant: variant.to_string(),
            lib_path: lib_path.to_path_buf(),
            libtorch_version: PYTORCH_VERSION.to_string(),
        }),
    }
}

fn prompt_backend() -> Result<String> {
    eprintln!("Select backend:");
    eprintln!("  [1] candle  — pure Rust, no external dependencies (default)");
    eprintln!("  [2] tch     — LibTorch C++ backend, faster for large models");
    eprint!("\nChoice [1]: ");
    std::io::stderr().flush()?;
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    match input.trim() {
        "" | "1" | "candle" => Ok("candle".into()),
        "2" | "tch" => Ok("tch".into()),
        other => bail!("Invalid choice: {other}"),
    }
}

fn prompt_variant() -> Result<String> {
    eprintln!("\nSelect compute variant:");
    for (i, &(name, label, _)) in VARIANTS.iter().enumerate() {
        let default = if i == 0 { " (default)" } else { "" };
        eprintln!("  [{}] {:<8} — {}{}", i + 1, name, label, default);
    }
    eprint!("\nChoice [1]: ");
    std::io::stderr().flush()?;
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Ok(VARIANTS[0].0.to_string());
    }
    // Try as number
    if let Ok(n) = trimmed.parse::<usize>() {
        if n >= 1 && n <= VARIANTS.len() {
            return Ok(VARIANTS[n - 1].0.to_string());
        }
    }
    // Try as name
    for &(name, _, _) in VARIANTS {
        if trimmed.eq_ignore_ascii_case(name) {
            return Ok(name.to_string());
        }
    }
    bail!("Invalid choice: {trimmed}");
}

fn download_with_progress(url: &str) -> Result<Vec<u8>> {
    let resp = ureq::get(url)
        .call()
        .with_context(|| format!("Failed to fetch {url}"))?;

    let total = resp
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(0);

    let pb = if total > 0 {
        let pb = indicatif::ProgressBar::new(total);
        pb.set_style(
            indicatif::ProgressStyle::default_bar()
                .template("{spinner:.green} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
                .unwrap()
                .progress_chars("#>-"),
        );
        Some(pb)
    } else {
        eprintln!("Downloading (size unknown)...");
        None
    };

    let mut reader = resp.into_body().into_reader();
    let mut buf = Vec::with_capacity(total as usize);
    let mut chunk = [0u8; 64 * 1024];
    loop {
        let n = std::io::Read::read(&mut reader, &mut chunk)?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..n]);
        if let Some(ref pb) = pb {
            pb.set_position(buf.len() as u64);
        }
    }

    if let Some(pb) = pb {
        pb.finish_with_message("Download complete");
    }
    eprintln!("Downloaded {} bytes.", buf.len());
    Ok(buf)
}

fn extract_zip(data: &[u8], dest: &Path) -> Result<()> {
    let cursor = std::io::Cursor::new(data);
    let mut archive = zip::ZipArchive::new(cursor)?;
    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        let outpath = dest.join(file.mangled_name());
        if file.is_dir() {
            fs::create_dir_all(&outpath)?;
        } else {
            if let Some(parent) = outpath.parent() {
                fs::create_dir_all(parent)?;
            }
            let mut outfile = fs::File::create(&outpath)?;
            std::io::copy(&mut file, &mut outfile)?;
            // Preserve executable permissions on Unix
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                if let Some(mode) = file.unix_mode() {
                    fs::set_permissions(&outpath, fs::Permissions::from_mode(mode))?;
                }
            }
        }
    }
    Ok(())
}
