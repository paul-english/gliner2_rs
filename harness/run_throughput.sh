#!/usr/bin/env bash
# Throughput benchmark (64 samples by default). Not invoked from CI.
#
# Default: one Rust backend (candle or tch via GLINER2_THROUGHPUT_BACKEND).
# Set GLINER2_BENCH_TCH=1 to time both Candle and tch-rs in one run; tch builds use
# --features tch-backend,download-libtorch (torch-sys downloads a matching LibTorch).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FIXTURES="${1:-$ROOT/harness/fixtures.json}"
RUST_SEQ_OUT="${2:-${TMPDIR:-/tmp}/gliner2_rust_throughput_seq.json}"
RUST_BATCH_OUT="${3:-${TMPDIR:-/tmp}/gliner2_rust_throughput_batch.json}"
SAMPLES="${4:-64}"

export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target}"
# shellcheck source=harness/prepend_libtorch_ld_path.sh
. "$ROOT/harness/prepend_libtorch_ld_path.sh"
PY_OUT="${TMPDIR:-/tmp}/gliner2_python_throughput.json"

BIN="$CARGO_TARGET_DIR/release/harness_throughput"
RUST_SEQ_TCH="${RUST_SEQ_OUT%.json}_tch.json"
RUST_BATCH_TCH="${RUST_BATCH_OUT%.json}_tch.json"

if [[ "${GLINER2_BENCH_TCH:-}" == "1" ]]; then
  (cd "$ROOT" && cargo build -q --release -p harness_compare --features tch-backend,download-libtorch --bin harness_throughput \
    && "$BIN" "$FIXTURES" --backend candle --samples "$SAMPLES" --rust-batch-size 1 >"$RUST_SEQ_OUT" \
    && "$BIN" "$FIXTURES" --backend candle --samples "$SAMPLES" --rust-batch-size "$SAMPLES" >"$RUST_BATCH_OUT" \
    && "$BIN" "$FIXTURES" --backend tch --samples "$SAMPLES" --rust-batch-size 1 >"$RUST_SEQ_TCH" \
    && "$BIN" "$FIXTURES" --backend tch --samples "$SAMPLES" --rust-batch-size "$SAMPLES" >"$RUST_BATCH_TCH")
  (cd "$ROOT/harness" && env CUDA_VISIBLE_DEVICES= uv run python benchmark_throughput.py \
    --fixtures "$FIXTURES" --device cpu --mode both --samples "$SAMPLES") >"$PY_OUT"
  (cd "$ROOT/harness" && uv run python compare_throughput.py "$RUST_SEQ_OUT" "$RUST_BATCH_OUT" "$PY_OUT" \
    --rust-seq-tch "$RUST_SEQ_TCH" --rust-batch-tch "$RUST_BATCH_TCH")
  exit 0
fi

BACKEND="${GLINER2_THROUGHPUT_BACKEND:-candle}"
CARGO_FEATS=()
BACKEND_ARGS=()
if [[ "$BACKEND" == "tch" ]]; then
  CARGO_FEATS=(--features tch-backend,download-libtorch)
  BACKEND_ARGS=(--backend tch)
else
  BACKEND_ARGS=(--backend candle)
fi

(cd "$ROOT" && cargo build -q --release -p harness_compare "${CARGO_FEATS[@]}" --bin harness_throughput \
  && "$BIN" "$FIXTURES" "${BACKEND_ARGS[@]}" --samples "$SAMPLES" --rust-batch-size 1 >"$RUST_SEQ_OUT" \
  && "$BIN" "$FIXTURES" "${BACKEND_ARGS[@]}" --samples "$SAMPLES" --rust-batch-size "$SAMPLES" >"$RUST_BATCH_OUT")

(cd "$ROOT/harness" && env CUDA_VISIBLE_DEVICES= uv run python benchmark_throughput.py \
  --fixtures "$FIXTURES" --device cpu --mode both --samples "$SAMPLES") >"$PY_OUT"

(cd "$ROOT/harness" && uv run python compare_throughput.py "$RUST_SEQ_OUT" "$RUST_BATCH_OUT" "$PY_OUT")
