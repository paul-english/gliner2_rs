---
id: gli-4ck0
status: open
deps: []
links: []
created: 2026-04-05T15:49:37Z
type: feature
priority: 1
assignee: Paul English
tags: [concurrency, performance, batch-api, rayon]
---
# Inference pipeline concurrency (batch + CPU post-forward)

Add structured parallelism across the GLiNER2 Rust inference path so CPU-bound work scales with batch size and workload. Depends on introducing a batch API for the tensor forward path; until then this ticket defines specification and implementation order.

**Context:** The codebase today runs post-forward decoding (per-task extraction, span scoring, format_results) sequentially with no rayon/thread pool. Candle may parallelize some tensor kernels internally; explicit Rust-side concurrency is not used.

**Goal:** Exploit all worthwhile concurrency with clear priority on the largest wins first.

## Design

**Priority order (biggest wins first):**

### 1. Across batch records (primary)
After a **batched** encode/heads pass, per-record CPU work is largely independent: different `PreprocessedInput`, span grids, JSON fragments, shared read-only metadata (`ExtractionMetadata`, schema).

- **Pattern:** One batched model forward → split outputs into per-index views (row slices, per-row tensors, or host-side buffers) → **parallel** per-record pipeline (e.g. Rayon `par_iter` over batch index).
- **Scope per record:** Gather per-task schema embeddings from `last_hidden`, reuse or slice `span_rep` per row, run existing `extract_*` / decode logic, `format_results` for that row only.
- **API shape:** Introduce something like `extract_batch(...)` where the hot path is `fn process_record(i) -> Result<Value>` with **no shared mutable map** across `i`; merge outputs in **deterministic index order**.
- **Why first:** Cost scales ~linearly with batch size; minimal synchronization (join at end); matches the main throughput story once a batch API exists.

### 2. Parallel preprocess (before forward)
Tokenization, schema transform, and building `PreprocessedInput` are CPU-heavy and independent across records.

- Run preprocess in parallel (same thread pool as post or one global pool), then collate/pad for batched forward.
- Reduces GPU idle time if preprocess dominates.

### 3. Within one record: multi-task parallelism (secondary)
Tasks currently fill one `raw` map sequentially. Optional parallelism across `TaskType` branches:

- **Approach:** Each task produces a partial `Map<String, Value>` or `(key, value)` list, then **merge** (keys should be disjoint).
- **When useful:** Small batch sizes but many tasks; or to soak CPU when batch-parallelism is underutilized.
- **Amdahl:** Few tasks per record vs many records → lower total gain than (1).

### 4. Inner loops (last / profiling-driven)
Nested span-grid loops, `tensor_to_vec4`-style walks: parallelize only if profiling shows a **large** hot loop **after** (1)–(2) are in place. Rayon overhead often dominates small inner chunks; tensor-heavy paths already benefit from BLAS/SIMD.

## Architecture boundary
- **Batched tensor stage:** GPU/batched kernels — not replaced by Rayon over spans.
- **CPU stage:** Explicit parallel boundary **after** batched forward: `split_batched_outputs` → `par_iter` over records.

## Implementation notes
- **Dependencies:** Add `rayon` (or equivalent) when implementing; keep optional `features` if some targets need single-threaded determinism.
- **Send/Sync:** Verify `candle_core::Tensor` (and views) are safe to use per-thread on target `Device` (CPU vs CUDA). Prefer per-row **clones/slices** or **host copies** at the parallel boundary if GPU handles are awkward.
- **Determinism:** Parallel float reductions can vary by schedule; for reproducible CI/benchmarks, document `RAYON_NUM_THREADS` or fixed pool + ordered reduction.
- **Pool sizing:** One global Rayon pool for preprocess + post; avoid unbounded nested parallelism.

## Relation to current code
- `extract_with_schema` (`src/extract.rs`): sequential `for` over `task_types`; future batch path should lift **outer** parallelism to batch index, not only inner task loops.
- `model.rs`: batch size 1 assumptions today; batch API must feed row-aligned hidden states and span reps into per-record CPU workers.

## References
- Prior discussion: prioritize batch-dimension parallelism, then preprocess, then per-task, then inner loops only if profiled.

## Acceptance Criteria

- Documented prioritized concurrency plan: batch records → parallel preprocess → optional per-task parallelism → inner loops only if profiled.
- Clear API boundary: batched tensor work vs parallel per-record CPU post-processing with deterministic ordered merge.
- Implementation constraints captured: `Send`/`Sync` / per-row tensor handoff, optional Rayon `feature`, determinism (`RAYON_NUM_THREADS` / ordered reduce), single pool / no nested `par_iter` storms.
- Follow-up work (batch API implementation, Rayon wiring, benchmarks) tracked in separate tickets with deps/links when created.
