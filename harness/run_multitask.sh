#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FIXTURES="${1:-$ROOT/harness/fixtures_multitask.json}"
RUST_OUT="${2:-${TMPDIR:-/tmp}/gliner2_rust_mt.json}"
PY_OUT="${3:-${TMPDIR:-/tmp}/gliner2_python_mt.json}"

export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target}"

(cd "$ROOT" && cargo build -q --release -p harness_compare --bin harness_compare_mt \
  && "$CARGO_TARGET_DIR/release/harness_compare_mt" "$FIXTURES") >"$RUST_OUT"
(cd "$ROOT/harness" && uv run python run_multitask_python.py --fixtures "$FIXTURES") >"$PY_OUT"
(cd "$ROOT/harness" && uv run python compare_mt.py "$RUST_OUT" "$PY_OUT")
