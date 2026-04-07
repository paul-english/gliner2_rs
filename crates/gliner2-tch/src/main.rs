use anyhow::Result;
use clap::Parser;

fn main() -> Result<()> {
    let cli = gliner2::cli::Cli::parse();

    let level = match cli.log_level.to_lowercase().as_str() {
        "off" => tracing::Level::ERROR,
        "error" => tracing::Level::ERROR,
        "warn" => tracing::Level::WARN,
        "info" => tracing::Level::INFO,
        "debug" => tracing::Level::DEBUG,
        "trace" => tracing::Level::TRACE,
        _ => tracing::Level::INFO,
    };
    tracing_subscriber::fmt().with_max_level(level).init();

    let backend = cli.backend.clone().unwrap_or_else(|| "tch".into());
    gliner2::cli::run(cli, &backend)
}
