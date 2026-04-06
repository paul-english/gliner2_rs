#!/usr/bin/env bash
# Run entity compare, multitask compare, and (optionally) throughput in one flow.
# By default runs all three backends: Rust Candle, Rust tch-rs (LibTorch), and Python.
# Artifacts go to harness/.compare_last/ for README patching and inspection.
#
# Usage:
#   bash harness/run_compare_all.sh [--candle-only] [--skip-throughput] [--update-readme] [entity_fixtures] [mt_fixtures] [throughput_samples]
#
# Environment (passed through to child scripts):
#   GLINER2_BENCH_TCH — set to 1 by this script unless --candle-only (then unset/0).
#   GLINER2_COMPARE_WARN_ONLY, GLINER2_THROUGHPUT_BACKEND, CARGO_TARGET_DIR, etc.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ARTIFACT_DIR="${ROOT}/harness/.compare_last"
SKIP_THROUGHPUT=0
UPDATE_README=0
CANDLE_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --candle-only)
      CANDLE_ONLY=1
      shift
      ;;
    --skip-throughput)
      SKIP_THROUGHPUT=1
      shift
      ;;
    --update-readme)
      UPDATE_README=1
      shift
      ;;
    -*)
      echo "unknown option: $1" >&2
      exit 2
      ;;
    *)
      break
      ;;
  esac
done

ENTITY_FIX="${1:-$ROOT/harness/fixtures.json}"
MT_FIX="${2:-$ROOT/harness/fixtures_multitask.json}"
SAMPLES="${3:-64}"

mkdir -p "$ARTIFACT_DIR"

if [[ "$CANDLE_ONLY" -eq 1 ]]; then
  export GLINER2_BENCH_TCH=0
else
  # Entity + multitask + throughput: Rust Candle, Rust tch-rs, and Python.
  export GLINER2_BENCH_TCH=1
fi

if [[ "$CANDLE_ONLY" -eq 1 ]]; then
  echo "==> Backends: Candle + Python only (--candle-only; skipping tch-rs)"
else
  echo "==> Backends: Candle + tch-rs + Python (GLINER2_BENCH_TCH=1)"
fi
echo "==> Entity harness (fixtures: $ENTITY_FIX)"
bash "$ROOT/harness/run_all.sh" "$ENTITY_FIX" "$ARTIFACT_DIR/entity_rust.json" "$ARTIFACT_DIR/entity_python.json"

echo "==> Multitask harness (fixtures: $MT_FIX)"
bash "$ROOT/harness/run_multitask.sh" "$MT_FIX" "$ARTIFACT_DIR/mt_rust.json" "$ARTIFACT_DIR/mt_python.json"

if [[ "$SKIP_THROUGHPUT" -eq 0 ]]; then
  echo "==> Throughput (fixtures: $ENTITY_FIX, samples: $SAMPLES)"
  bash "$ROOT/harness/run_throughput.sh" "$ENTITY_FIX" \
    "$ARTIFACT_DIR/throughput_rust_seq_candle.json" \
    "$ARTIFACT_DIR/throughput_rust_batch_8_candle.json" \
    "$ARTIFACT_DIR/throughput_rust_batch_64_candle.json" \
    "$SAMPLES" \
    "$ARTIFACT_DIR/throughput_python.json"
else
  echo "==> Skipping throughput (--skip-throughput)"
fi

if [[ "$UPDATE_README" -eq 1 ]]; then
  echo "==> Patching README.md"
  # Use ./patch_readme.py so uv/python resolve the harness script, not another module named patch_readme.
  (cd "$ROOT/harness" && uv run python ./patch_readme.py --readme "$ROOT/README.md" --artifact-dir "$ARTIFACT_DIR")
fi

echo "Done. Artifacts: $ARTIFACT_DIR"
