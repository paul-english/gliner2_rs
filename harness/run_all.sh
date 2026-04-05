#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FIXTURES="${1:-$ROOT/harness/fixtures.json}"
RUST_OUT="${2:-${TMPDIR:-/tmp}/gliner2_rust_harness.json}"
PY_OUT="${3:-${TMPDIR:-/tmp}/gliner2_python_harness.json}"

# Pin target dir to the repo so the binary path is stable (some environments override CARGO_TARGET_DIR).
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target}"

# Build workspace member quietly; stdout only is JSON (stderr may show compile progress).
(cd "$ROOT" && cargo build -q --release -p harness_compare && "$CARGO_TARGET_DIR/release/harness_compare" "$FIXTURES") >"$RUST_OUT"
# CPU-vs-CPU: hide discrete GPUs from PyTorch and load weights onto CPU.
(cd "$ROOT/harness" && env CUDA_VISIBLE_DEVICES= uv run python run_python.py --fixtures "$FIXTURES" --device cpu) >"$PY_OUT"
(cd "$ROOT/harness" && uv run python compare.py "$RUST_OUT" "$PY_OUT")
