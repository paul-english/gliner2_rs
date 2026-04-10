---
id: gli-a4wz
status: closed
deps: []
links: []
created: 2026-04-10T01:09:28Z
type: bug
priority: 1
assignee: Paul English
tags: [ci, harness, tch]
---
# Fix harness libtorch runtime path in CI

Harness scripts source prepend_libtorch_ld_path.sh before the tch download-libtorch build, so clean CI jobs launch tch binaries without LD_LIBRARY_PATH set and fail to load libtorch_cpu.so.


## Notes

**2026-04-10T01:13:27Z**

Root cause: harness scripts sourced prepend_libtorch_ld_path.sh before the tch download-libtorch build. In clean CI, target/release/build/torch-sys-*/out does not exist yet, so LD_LIBRARY_PATH remains empty and the later tch binary fails to load libtorch_cpu.so. Fixed by refreshing the helper after tch builds in run_all.sh, run_multitask.sh, and run_throughput.sh. Verified with bash -n; local end-to-end tch build is blocked by a separate host PyTorch 2.10.0 vs tch 2.11.0 mismatch.
