#!/usr/bin/env bash
# Throughput benchmark (64 samples by default). Not invoked from CI.
#
# Default: one Rust backend (candle or tch via GLINER2_THROUGHPUT_BACKEND).
# Set GLINER2_BENCH_TCH=1 to time both Candle and tch-rs in one run; tch builds use
# --features tch-backend,download-libtorch (torch-sys downloads a matching LibTorch).
#
# Usage:
#   bash harness/run_throughput.sh [fixtures.json] [rust_seq_out.json] [rust_batch_8_out.json] [rust_batch_64_out.json] [samples] [python_out.json]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FIXTURES="${1:-$ROOT/harness/fixtures.json}"
RUST_SEQ_OUT="${2:-${TMPDIR:-/tmp}/gliner2_rust_throughput_seq.json}"
RUST_BATCH_8_OUT="${3:-${TMPDIR:-/tmp}/gliner2_rust_throughput_batch8.json}"
RUST_BATCH_64_OUT="${4:-${TMPDIR:-/tmp}/gliner2_rust_throughput_batch64.json}"
SAMPLES="${5:-64}"
PY_OUT="${6:-${TMPDIR:-/tmp}/gliner2_python_throughput.json}"

export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target}"
# shellcheck source=harness/prepend_libtorch_ld_path.sh
. "$ROOT/harness/prepend_libtorch_ld_path.sh"

BIN="$CARGO_TARGET_DIR/release/harness_throughput"

rust_tch_path() {
  local base="$1"
  echo "${base%.json}_tch.json"
}

# Run sequential (bs=1) plus batched at 8 and 64 for one Rust backend (candle or tch).
run_rust_triple() {
  local backend_name="$1"
  local seq_out="$2"
  local b8_out="$3"
  local b64_out="$4"
  (cd "$ROOT" && \
    "$BIN" "$FIXTURES" --backend "$backend_name" --samples "$SAMPLES" --rust-batch-size 1 >"$seq_out" \
    && "$BIN" "$FIXTURES" --backend "$backend_name" --samples "$SAMPLES" --rust-batch-size 8 >"$b8_out" \
    && "$BIN" "$FIXTURES" --backend "$backend_name" --samples "$SAMPLES" --rust-batch-size 64 >"$b64_out")
}

if [[ "${GLINER2_BENCH_TCH:-}" == "1" ]]; then
  (cd "$ROOT" && cargo build -q --release -p harness_compare --features tch-backend,download-libtorch --bin harness_throughput)
  run_rust_triple candle "$RUST_SEQ_OUT" "$RUST_BATCH_8_OUT" "$RUST_BATCH_64_OUT"
  RUST_SEQ_TCH="$(rust_tch_path "$RUST_SEQ_OUT")"
  RUST_BATCH_8_TCH="$(rust_tch_path "$RUST_BATCH_8_OUT")"
  RUST_BATCH_64_TCH="$(rust_tch_path "$RUST_BATCH_64_OUT")"
  run_rust_triple tch "$RUST_SEQ_TCH" "$RUST_BATCH_8_TCH" "$RUST_BATCH_64_TCH"
  (cd "$ROOT/harness" && env CUDA_VISIBLE_DEVICES= uv run python benchmark_throughput.py \
    --fixtures "$FIXTURES" --device cpu --mode both --samples "$SAMPLES" \
    --batched-batch-sizes 8,64) >"$PY_OUT"
  (cd "$ROOT/harness" && uv run python compare_throughput.py \
    "$RUST_SEQ_OUT" "$RUST_BATCH_8_OUT" "$RUST_BATCH_64_OUT" "$PY_OUT" \
    --rust-seq-tch "$RUST_SEQ_TCH" \
    --rust-batch-8-tch "$RUST_BATCH_8_TCH" \
    --rust-batch-64-tch "$RUST_BATCH_64_TCH")
  exit 0
fi

BACKEND="${GLINER2_THROUGHPUT_BACKEND:-candle}"
CARGO_FEATS=()
if [[ "$BACKEND" == "tch" ]]; then
  CARGO_FEATS=(--features tch-backend,download-libtorch)
fi

(cd "$ROOT" && cargo build -q --release -p harness_compare "${CARGO_FEATS[@]}" --bin harness_throughput \
  && run_rust_triple "$BACKEND" "$RUST_SEQ_OUT" "$RUST_BATCH_8_OUT" "$RUST_BATCH_64_OUT")

(cd "$ROOT/harness" && env CUDA_VISIBLE_DEVICES= uv run python benchmark_throughput.py \
  --fixtures "$FIXTURES" --device cpu --mode both --samples "$SAMPLES" \
  --batched-batch-sizes 8,64) >"$PY_OUT"

(cd "$ROOT/harness" && uv run python compare_throughput.py \
  "$RUST_SEQ_OUT" "$RUST_BATCH_8_OUT" "$RUST_BATCH_64_OUT" "$PY_OUT")
