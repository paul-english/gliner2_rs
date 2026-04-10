#!/usr/bin/env bash
# Set GLINER2_BENCH_TCH=1 to also run harness_compare_mt --backend tch (build: tch-backend,download-libtorch).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FIXTURES="${1:-$ROOT/harness/fixtures_multitask.json}"
RUST_OUT="${2:-${TMPDIR:-/tmp}/gliner2_rust_mt.json}"
PY_OUT="${3:-${TMPDIR:-/tmp}/gliner2_python_mt.json}"
RUST_TCH_OUT="${RUST_TCH_OUT:-${RUST_OUT%.json}_tch.json}"

export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target}"
refresh_libtorch_ld_path() {
  # shellcheck source=harness/prepend_libtorch_ld_path.sh
  . "$ROOT/harness/prepend_libtorch_ld_path.sh"
}

refresh_libtorch_ld_path

(cd "$ROOT" && cargo build -q --release -p harness_compare --bin harness_compare_mt \
  && "$CARGO_TARGET_DIR/release/harness_compare_mt" "$FIXTURES") >"$RUST_OUT"
COMPARE_EXTRA=()
if [[ "${GLINER2_BENCH_TCH:-}" == "1" ]]; then
  (cd "$ROOT" && cargo build -q --release -p harness_compare --features tch-backend,download-libtorch --bin harness_compare_mt)
  refresh_libtorch_ld_path
  (cd "$ROOT" && "$CARGO_TARGET_DIR/release/harness_compare_mt" "$FIXTURES" --backend tch) >"$RUST_TCH_OUT"
  COMPARE_EXTRA=(--rust-tch "$RUST_TCH_OUT")
fi
(cd "$ROOT/harness" && env CUDA_VISIBLE_DEVICES= uv run python run_multitask_python.py --fixtures "$FIXTURES" --device cpu) >"$PY_OUT"
WARN=()
[[ "${GLINER2_COMPARE_WARN_ONLY:-}" == "1" ]] && WARN=(--warn-only)
(cd "$ROOT/harness" && uv run python compare_mt.py "$RUST_OUT" "$PY_OUT" "${COMPARE_EXTRA[@]}" "${WARN[@]}")
