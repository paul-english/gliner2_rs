---
id: gli-wid9
status: open
deps: []
links: []
created: 2026-04-08T01:20:11Z
type: task
priority: 2
assignee: Paul English
---
# Enable release build on OSX

https://github.com/paul-english/gliner2_rs/actions/runs/24090407443/job/70274906361
```
error: failed to run custom build command for `torch-sys v0.24.0`

Caused by:
  process didn't exit successfully: `/Users/runner/work/gliner2_rs/gliner2_rs/target/release/build/torch-sys-b92cc603f6298cdb/build-script-build` (exit status: 101)
  --- stdout
  cargo:rerun-if-env-changed=LIBTORCH_USE_PYTORCH
  cargo:rerun-if-env-changed=LIBTORCH
  cargo:rerun-if-env-changed=TORCH_CUDA_VERSION

  --- stderr

  thread 'main' (44682) panicked at /Users/runner/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/torch-sys-0.24.0/build.rs:325:65:
  Failed to retrieve torch from pypi.  Pre-built version of libtorch for apple silicon are not available.
                              You can install torch manually following the indications from https://github.com/LaurentMazare/tch-rs/issues/629
                              pip3 install torch=={TORCH_VERSION}
                              Then update the following environment variables:
                              export LIBTORCH=$(python3 -c 'import torch; from pathlib import Path; print(Path(torch.__file__).parent)')
                              export DYLD_LIBRARY_PATH=${{LIBTORCH}}/lib
                              : Failed to find arm64 macosx wheel from pypi
  note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
warning: build failed, waiting for other jobs to finish...
```
