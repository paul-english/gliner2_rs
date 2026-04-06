// DeBERTa V2 model internalized from rust-bert 0.23.
// Original: https://github.com/guillaume-be/rust-bert
// License: Apache-2.0

// Ported upstream code — suppress style lints rather than rewriting.
#[allow(dead_code, clippy::assign_op_pattern, clippy::manual_contains)]
mod attention;
#[allow(dead_code, clippy::assign_op_pattern)]
mod common;
#[allow(dead_code, clippy::manual_contains)]
mod config;
#[allow(dead_code, clippy::assign_op_pattern)]
mod encoder;

pub use config::DebertaV2Config;
pub use encoder::DebertaV2Model;
