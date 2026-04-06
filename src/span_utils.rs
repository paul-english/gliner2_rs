//! Shared CPU utilities for span index generation, used by both candle and tch backends.

/// Generate span index pairs `(start, end)` for a text of `text_len` words with up to
/// `max_width` span width. Returns `text_len * max_width` pairs.
pub fn generate_span_indices(text_len: usize, max_width: usize) -> Vec<[usize; 2]> {
    let mut indices = Vec::with_capacity(text_len * max_width);
    for i in 0..text_len {
        for w in 0..max_width {
            let end = (i + w).min(text_len.saturating_sub(1));
            indices.push([i, end]);
        }
    }
    indices
}

/// Generate batched span indices for variable-length texts padded to `max_text_len`.
///
/// Returns a flat buffer of `batch_size * max_text_len * max_width` pairs.
/// Invalid spans (where `start + width >= text_length`) are set to `[0, 0]`.
pub fn generate_batched_span_indices(
    text_lengths: &[usize],
    max_text_len: usize,
    max_width: usize,
) -> Vec<[usize; 2]> {
    let n_spans = max_text_len * max_width;
    let batch_size = text_lengths.len();
    let mut result = vec![[0usize; 2]; batch_size * n_spans];
    for (b, &tl) in text_lengths.iter().enumerate() {
        for i in 0..max_text_len {
            for w in 0..max_width {
                let idx = i * max_width + w;
                let end = i + w;
                if end < tl {
                    result[b * n_spans + idx] = [i, end];
                }
            }
        }
    }
    result
}
