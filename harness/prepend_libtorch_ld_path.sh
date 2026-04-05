# Extend the dynamic loader path for torch-sys builds that use download-libtorch.
# Expects CARGO_TARGET_DIR and a release build under .../release/build/torch-sys-*/out/...
# shellcheck shell=bash
[[ -n "${CARGO_TARGET_DIR:-}" ]] || return 0
_gliner2_lt="$(
  find "$CARGO_TARGET_DIR/release/build" -path '*/torch-sys-*/out/*' \( -name 'libtorch_cpu.so' -o -name 'libtorch_cpu.dylib' \) -print -quit 2>/dev/null
)"
if [[ -n "$_gliner2_lt" ]]; then
  _gliner2_dir="$(dirname "$_gliner2_lt")"
  export LD_LIBRARY_PATH="${_gliner2_dir}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export DYLD_LIBRARY_PATH="${_gliner2_dir}${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
  unset _gliner2_dir
fi
unset _gliner2_lt
