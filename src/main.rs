use anyhow::Result;
use clap::{Parser, Subcommand};
use gliner2::cli;
use std::path::PathBuf;

/// Launcher-only subcommands (setup, status). Inference commands are handled
/// by the full `cli::Cli` parser when these don't match.
#[derive(Parser)]
#[command(
    name = "gliner2",
    version,
    about = "Gliner2 CLI",
    disable_help_subcommand = true
)]
struct LauncherCli {
    #[command(subcommand)]
    command: Option<LauncherCommands>,
}

#[derive(Subcommand)]
enum LauncherCommands {
    /// Configure backend and download libtorch
    Setup {
        /// Skip interactive prompts
        #[arg(long)]
        non_interactive: bool,
        /// Backend: candle or tch
        #[arg(long)]
        backend: Option<String>,
        /// LibTorch variant: cpu, cu128, cu130, rocm62
        #[arg(long)]
        variant: Option<String>,
    },
    /// Show current configuration and backend status
    Status,
}

fn main() -> Result<()> {
    // Two-phase parse: try launcher commands first, then fall through to full inference CLI.
    // We check the raw args to avoid clap errors on inference subcommands like "entities".
    let first_positional = std::env::args().skip(1).find(|a| !a.starts_with('-'));

    match first_positional.as_deref() {
        Some("setup") => {
            let launcher = LauncherCli::parse();
            if let Some(LauncherCommands::Setup {
                non_interactive,
                backend,
                variant,
            }) = launcher.command
            {
                return gliner2::setup::run_setup(
                    non_interactive,
                    backend.as_deref(),
                    variant.as_deref(),
                );
            }
            unreachable!();
        }
        Some("status") => {
            return show_status();
        }
        _ => {}
    }

    // Inference path: parse with the full CLI struct.
    let cli_args = cli::Cli::parse();

    // Resolve backend: --backend flag > GLINER2_BACKEND env > config.toml > "candle"
    let backend = match &cli_args.backend {
        Some(b) => b.clone(),
        None => {
            if let Ok(Some(cfg)) = gliner2::setup::load_config() {
                cfg.backend.default.clone()
            } else {
                "candle".to_string()
            }
        }
    };

    init_tracing(&cli_args.log_level);

    if backend == "tch" {
        #[cfg(feature = "tch")]
        {
            return cli::run(cli_args, &backend);
        }

        #[cfg(not(feature = "tch"))]
        {
            return exec_tch_binary();
        }
    }

    cli::run(cli_args, &backend)
}

fn init_tracing(log_level: &str) {
    let level = match log_level.to_lowercase().as_str() {
        "off" => tracing::Level::ERROR,
        "error" => tracing::Level::ERROR,
        "warn" => tracing::Level::WARN,
        "info" => tracing::Level::INFO,
        "debug" => tracing::Level::DEBUG,
        "trace" => tracing::Level::TRACE,
        _ => tracing::Level::INFO,
    };
    tracing_subscriber::fmt().with_max_level(level).init();
}

fn show_status() -> Result<()> {
    let candle_available = cfg!(feature = "candle");
    let tch_compiled = cfg!(feature = "tch");

    eprintln!("gliner2 status\n");
    eprintln!("Compiled backends:");
    eprintln!("  candle: {}", if candle_available { "yes" } else { "no" });
    eprintln!(
        "  tch:    {}",
        if tch_compiled {
            "yes (compiled-in)"
        } else {
            "no (uses gliner2-tch binary)"
        }
    );

    match gliner2::setup::load_config()? {
        Some(cfg) => {
            eprintln!("\nConfig: {}", gliner2::setup::config_path()?.display());
            eprintln!("  default backend: {}", cfg.backend.default);
            if let Some(tch) = &cfg.tch {
                eprintln!("  tch variant:     {}", tch.variant);
                eprintln!("  libtorch path:   {}", tch.lib_path.display());
                eprintln!("  libtorch ver:    {}", tch.libtorch_version);
                let lib_exists = tch.lib_path.exists();
                eprintln!(
                    "  libtorch present: {}",
                    if lib_exists {
                        "yes"
                    } else {
                        "NO — run `gliner2 setup`"
                    }
                );
            }
        }
        None => {
            eprintln!("\nNo config found. Run `gliner2 setup` to configure.");
        }
    }

    if !tch_compiled {
        let found = find_tch_binary().is_some();
        eprintln!(
            "\ngliner2-tch binary: {}",
            if found {
                "found"
            } else {
                "not found — install with `cargo binstall gliner2-tch`"
            }
        );
    }

    Ok(())
}

/// Find the gliner2-tch binary: same directory as current exe, then PATH.
fn find_tch_binary() -> Option<PathBuf> {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let candidate = dir.join("gliner2-tch");
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }

    if let Some(paths) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&paths) {
            let candidate = dir.join("gliner2-tch");
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }

    None
}

#[cfg(not(feature = "tch"))]
fn exec_tch_binary() -> Result<()> {
    let config = gliner2::setup::load_config()?;
    let lib_path = config
        .as_ref()
        .and_then(|c| c.tch.as_ref())
        .map(|t| t.lib_path.clone());

    let lib_path = match lib_path {
        Some(lp) if lp.exists() => lp,
        _ => {
            anyhow::bail!(
                "LibTorch not configured. Run `gliner2 setup` to download and configure LibTorch."
            );
        }
    };

    let tch_bin = find_tch_binary().ok_or_else(|| {
        anyhow::anyhow!(
            "Backend \"tch\" requested but gliner2-tch binary not found.\n\
             Install it with: cargo binstall gliner2-tch\n\
             For development, use: just run-tch <args>"
        )
    })?;

    let existing = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();
    let new_ld = if existing.is_empty() {
        lib_path.to_string_lossy().to_string()
    } else {
        format!("{}:{existing}", lib_path.display())
    };

    let existing_dyld = std::env::var("DYLD_LIBRARY_PATH").unwrap_or_default();
    let new_dyld = if existing_dyld.is_empty() {
        lib_path.to_string_lossy().to_string()
    } else {
        format!("{}:{existing_dyld}", lib_path.display())
    };

    let args: Vec<String> = std::env::args().skip(1).collect();

    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        let err = std::process::Command::new(&tch_bin)
            .args(&args)
            .env("LD_LIBRARY_PATH", &new_ld)
            .env("DYLD_LIBRARY_PATH", &new_dyld)
            .exec();
        anyhow::bail!("Failed to exec {}: {}", tch_bin.display(), err);
    }

    #[cfg(not(unix))]
    {
        let status = std::process::Command::new(&tch_bin)
            .args(&args)
            .env("LD_LIBRARY_PATH", &new_ld)
            .env("DYLD_LIBRARY_PATH", &new_dyld)
            .status()?;
        std::process::exit(status.code().unwrap_or(1));
    }
}
