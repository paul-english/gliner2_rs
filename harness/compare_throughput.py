#!/usr/bin/env python3
"""Compare throughput JSON: Rust sequential + Rust batched vs Python (from benchmark_throughput)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _f(x: Any) -> float:
    return float(x) if x is not None else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "rust_sequential_json",
        type=Path,
        help="harness_throughput --rust-batch-size 1",
    )
    ap.add_argument(
        "rust_batched_json",
        type=Path,
        help="harness_throughput --rust-batch-size N (typically N == samples)",
    )
    ap.add_argument("python_json", type=Path, help="benchmark_throughput.py --mode both")
    args = ap.parse_args()

    r_seq = json.loads(args.rust_sequential_json.read_text(encoding="utf-8"))
    r_bat = json.loads(args.rust_batched_json.read_text(encoding="utf-8"))
    py = json.loads(args.python_json.read_text(encoding="utf-8"))

    print("=== Throughput comparison (local benchmark; not run in CI) ===")
    print(
        f"model={r_seq.get('model_id')}  rust_device={r_seq.get('device_note')}  "
        f"py_device={py.get('device_note')}"
    )
    print(
        f"samples={r_seq.get('samples')}  warmup_full_passes={r_seq.get('warmup_full_passes')}"
    )
    print()
    print(f"{'':30} {'total_infer_ms':>14} {'samples/s':>12}  notes")
    print("-" * 70)

    rs_ms = _f(r_seq.get("total_infer_ms"))
    rs_sps = _f(r_seq.get("samples_per_sec"))
    print(
        f"{'Rust sequential (bs=1)':30} {rs_ms:14.3f} {rs_sps:12.3f}  {r_seq.get('mode', '')}"
    )

    if "sequential" in py:
        s = py["sequential"]
        p_ms = _f(s.get("total_infer_ms"))
        p_sps = _f(s.get("samples_per_sec"))
        ratio = p_ms / rs_ms if rs_ms > 0 else float("inf")
        print(
            f"{'Python sequential (bs=1)':30} {p_ms:14.3f} {p_sps:12.3f}  "
            f"python/rust {ratio:.3f}x"
        )

    rb_ms = _f(r_bat.get("total_infer_ms"))
    rb_sps = _f(r_bat.get("samples_per_sec"))
    rbs = r_bat.get("batch_size", "?")
    print(
        f"{f'Rust batched (bs={rbs})':30} {rb_ms:14.3f} {rb_sps:12.3f}  {r_bat.get('mode', '')}"
    )

    if "batched" in py:
        b = py["batched"]
        p_ms = _f(b.get("total_infer_ms"))
        p_sps = _f(b.get("samples_per_sec"))
        bs = b.get("batch_size", "?")
        ratio = p_ms / rb_ms if rb_ms > 0 else float("inf")
        print(
            f"{f'Python batched (bs={bs})':30} {p_ms:14.3f} {p_sps:12.3f}  "
            f"python/rust {ratio:.3f}x"
        )

    print()
    print(
        "load_model_ms  "
        f"rust_seq={_f(r_seq.get('load_model_ms')):.3f}  "
        f"rust_bat={_f(r_bat.get('load_model_ms')):.3f}  "
        f"python={_f(py.get('load_model_ms')):.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
