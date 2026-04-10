---
id: gli-jq1m
status: closed
deps: []
links: []
created: 2026-04-09T00:35:52Z
type: task
priority: 2
assignee: Paul English
---
# Support multiple GPUs


## Notes

**2026-04-10T01:27:08Z**

Implemented multi-CPU/GPU support with --num-workers flag:
- Added --num-workers CLI flag (default: 1)
- Created create_engines() function to instantiate multiple engine instances
- Implemented run_extract_streaming_multi() for parallel execution across workers
- For tch backend: automatically assigns different GPU devices (cuda:0, cuda:1, etc.) when multiple workers requested
- For candle backend: creates multiple CPU-bound engines for parallel processing
- Work is distributed across workers using rayon parallel iterators
- Results are collected and emitted in original order

Builds successfully with cargo build.

**2026-04-10T02:10:20Z**

Full implementation review completed. All three sub-tickets implemented:

1. gli-2pwt (--num-workers CLI flag): ✅
   - Added --num-workers flag with default 1
   - Dispatches to single-engine path when num_workers=1 for zero overhead
   - Dispatches to multi-engine parallel path when num_workers>1

2. gli-ky1e (multi-device tch backend): ✅
   - --device cuda auto-assigns cuda:0, cuda:1, etc. per worker
   - --device cuda:N assigns cuda:N, cuda:N+1, etc.
   - Non-indexed devices (cpu, mps, vulkan) share the same device across workers

3. gli-7so4 (parallel parquet I/O): ✅
   - Multi-file parquet glob processed in parallel with rayon
   - One engine per file (round-robin) to avoid nested parallelism
   - Single-file paths use multi-worker parallel extraction

Key design decisions:
- run_extract_dispatch() chooses single vs multi-engine path
- run_extract_streaming_multi() distributes batch_size chunks across workers
- Results are sorted and emitted in input order
- Each worker loads a full copy of model weights (documented in --help and README)
- Tracing log emitted when multi-worker mode is active
