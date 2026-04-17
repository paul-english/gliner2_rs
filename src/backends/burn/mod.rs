// Burn backend for GLiNER2 inference.
// Pure Rust (NdArray), no C/C++ dependencies.

#[allow(clippy::module_inception)]
mod deberta_v2;
mod layers;
mod model;
mod span_rep;
mod tensor_wrap;
mod weights;

pub use model::BurnExtractor;
