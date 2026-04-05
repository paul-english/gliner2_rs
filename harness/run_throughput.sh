#!/usr/bin/env bash
# Throughput benchmark (64 samples by default). Not invoked from CI.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FIXTURES="${1:-$ROOT/harness/fixtures.json}"
RUST_SEQ_OUT="${2:-${TMPDIR:-/tmp}/gliner2_rust_throughput_seq.json}"
RUST_BATCH_OUT="${3:-${TMPDIR:-/tmp}/gliner2_rust_throughput_batch.json}"
SAMPLES="${4:-64}"

export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target}"
PY_OUT="${TMPDIR:-/tmp}/gliner2_python_throughput.json"

BIN="$CARGO_TARGET_DIR/release/harness_throughput"

(cd "$ROOT" && cargo build -q --release -p harness_compare --bin harness_throughput \
  && "$BIN" "$FIXTURES" --samples "$SAMPLES" --rust-batch-size 1 >"$RUST_SEQ_OUT" \
  && "$BIN" "$FIXTURES" --samples "$SAMPLES" --rust-batch-size "$SAMPLES" >"$RUST_BATCH_OUT")

(cd "$ROOT/harness" && env CUDA_VISIBLE_DEVICES= uv run python benchmark_throughput.py \
  --fixtures "$FIXTURES" --device cpu --mode both --samples "$SAMPLES") >"$PY_OUT"

(cd "$ROOT/harness" && uv run python compare_throughput.py "$RUST_SEQ_OUT" "$RUST_BATCH_OUT" "$PY_OUT")
