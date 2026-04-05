#!/usr/bin/env bash
# Set GLINER2_BENCH_TCH=1 to also run harness_compare --backend tch (build: tch-backend,download-libtorch).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FIXTURES="${1:-$ROOT/harness/fixtures.json}"
RUST_OUT="${2:-${TMPDIR:-/tmp}/gliner2_rust_harness.json}"
PY_OUT="${3:-${TMPDIR:-/tmp}/gliner2_python_harness.json}"
RUST_TCH_OUT="${RUST_TCH_OUT:-${RUST_OUT%.json}_tch.json}"

# Pin target dir to the repo so the binary path is stable (some environments override CARGO_TARGET_DIR).
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target}"
# shellcheck source=harness/prepend_libtorch_ld_path.sh
. "$ROOT/harness/prepend_libtorch_ld_path.sh"

# Build workspace member quietly; stdout only is JSON (stderr may show compile progress).
(cd "$ROOT" && cargo build -q --release -p harness_compare && "$CARGO_TARGET_DIR/release/harness_compare" "$FIXTURES") >"$RUST_OUT"
COMPARE_EXTRA=()
if [[ "${GLINER2_BENCH_TCH:-}" == "1" ]]; then
  (cd "$ROOT" && cargo build -q --release -p harness_compare --features tch-backend,download-libtorch \
    && "$CARGO_TARGET_DIR/release/harness_compare" "$FIXTURES" --backend tch) >"$RUST_TCH_OUT"
  COMPARE_EXTRA=(--rust-tch "$RUST_TCH_OUT")
fi
# CPU-vs-CPU: hide discrete GPUs from PyTorch and load weights onto CPU.
(cd "$ROOT/harness" && env CUDA_VISIBLE_DEVICES= uv run python run_python.py --fixtures "$FIXTURES" --device cpu) >"$PY_OUT"
WARN=()
[[ "${GLINER2_COMPARE_WARN_ONLY:-}" == "1" ]] && WARN=(--warn-only)
(cd "$ROOT/harness" && uv run python compare.py "$RUST_OUT" "$PY_OUT" "${COMPARE_EXTRA[@]}" "${WARN[@]}")
