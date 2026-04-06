#!/usr/bin/env python3
"""Compare throughput JSON: Rust sequential + batched (8, 64) vs Python; optional second Rust (tch)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _f(x: Any) -> float:
    return float(x) if x is not None else 0.0


def _py_batched(py: dict[str, Any], bs: int) -> dict[str, Any] | None:
    by = py.get("batched_by_size")
    if isinstance(by, dict) and str(bs) in by:
        return by[str(bs)]
    # Legacy single batched block (batch_size == samples)
    b = py.get("batched")
    if b is not None and int(b.get("batch_size", -1)) == bs:
        return b
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "rust_sequential_json",
        type=Path,
        help="harness_throughput --rust-batch-size 1 --backend candle",
    )
    ap.add_argument(
        "rust_batch_8_json",
        type=Path,
        help="harness_throughput --rust-batch-size 8 --backend candle",
    )
    ap.add_argument(
        "rust_batch_64_json",
        type=Path,
        help="harness_throughput --rust-batch-size 64 --backend candle",
    )
    ap.add_argument(
        "python_json", type=Path, help="benchmark_throughput.py --mode both"
    )
    ap.add_argument(
        "--rust-seq-tch",
        type=Path,
        metavar="PATH",
        help="harness_throughput --backend tch --rust-batch-size 1",
    )
    ap.add_argument(
        "--rust-batch-8-tch",
        type=Path,
        metavar="PATH",
        help="harness_throughput --backend tch --rust-batch-size 8",
    )
    ap.add_argument(
        "--rust-batch-64-tch",
        type=Path,
        metavar="PATH",
        help="harness_throughput --backend tch --rust-batch-size 64",
    )
    args = ap.parse_args()

    try:
        r_seq = json.loads(args.rust_sequential_json.read_text(encoding="utf-8"))
        r_b8 = json.loads(args.rust_batch_8_json.read_text(encoding="utf-8"))
        r_b64 = json.loads(args.rust_batch_64_json.read_text(encoding="utf-8"))
        py = json.loads(args.python_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"error: failed to read JSON: {e}", file=sys.stderr)
        return 1

    r_seq_tch = (
        json.loads(args.rust_seq_tch.read_text(encoding="utf-8"))
        if args.rust_seq_tch
        else None
    )
    r_b8_tch = (
        json.loads(args.rust_batch_8_tch.read_text(encoding="utf-8"))
        if args.rust_batch_8_tch
        else None
    )
    r_b64_tch = (
        json.loads(args.rust_batch_64_tch.read_text(encoding="utf-8"))
        if args.rust_batch_64_tch
        else None
    )

    print("=== Throughput comparison (local benchmark; not run in CI) ===")
    rb = r_seq.get("backend", "candle")
    print(
        f"model={r_seq.get('model_id')}  rust_backend(candle)={rb}  "
        f"rust_device={r_seq.get('device_note')}  py_device={py.get('device_note')}"
    )
    if r_seq_tch and r_b8_tch and r_b64_tch:
        print(
            f"rust_backend(tch)={r_seq_tch.get('backend')}  "
            f"rust_tch_device={r_seq_tch.get('device_note')}"
        )
    print(
        f"samples={r_seq.get('samples')}  warmup_full_passes={r_seq.get('warmup_full_passes')}"
    )
    print()

    def row_tch(
        label: str,
        rs: dict[str, Any],
        rt: dict[str, Any],
        py_key_bs: int,
    ) -> None:
        rs_ms = _f(rs.get("total_infer_ms"))
        rt_ms = _f(rt.get("total_infer_ms"))
        pb = _py_batched(py, py_key_bs)
        if pb is None and py_key_bs == 1:
            s = py.get("sequential")
            p_ms = _f(s.get("total_infer_ms")) if s else 0.0
        elif pb is not None:
            p_ms = _f(pb.get("total_infer_ms"))
        else:
            p_ms = 0.0
        rs_sps = _f(rs.get("samples_per_sec"))
        rt_sps = _f(rt.get("samples_per_sec"))
        if py_key_bs == 1:
            ps = py.get("sequential")
            p_sps = _f(ps.get("samples_per_sec")) if ps else 0.0
        else:
            p_sps = _f(pb.get("samples_per_sec")) if pb else 0.0
        print(
            f"{label:36} {rs_ms:12.3f} {rs_sps:12.3f} {rt_ms:12.3f} {rt_sps:12.3f} "
            f"{p_ms:12.3f} {p_sps:12.3f} "
            f"{(p_ms / rs_ms if rs_ms else float('inf')):8.3f}x "
            f"{(rt_ms / rs_ms if rs_ms else float('inf')):8.3f}x "
            f"{(p_ms / rt_ms if rt_ms else float('inf')):8.3f}x"
        )

    if r_seq_tch and r_b8_tch and r_b64_tch:
        print(
            f"{'':36} {'cnd_ms':>12} {'cnd_s/s':>12} {'tch_ms':>12} {'tch_s/s':>12} "
            f"{'py_ms':>12} {'py_s/s':>12} {'py/cnd':>8} {'tch/cnd':>8} {'py/tch':>8}"
        )
        print("-" * 120)
        row_tch(
            "Sequential (bs=1)",
            r_seq,
            r_seq_tch,
            1,
        )
        row_tch(
            "Batched (bs=8)",
            r_b8,
            r_b8_tch,
            8,
        )
        row_tch(
            "Batched (bs=64)",
            r_b64,
            r_b64_tch,
            64,
        )
        print()
        print(
            "load_model_ms  "
            f"rust_candle_seq={_f(r_seq.get('load_model_ms')):.3f}  "
            f"rust_candle_b8={_f(r_b8.get('load_model_ms')):.3f}  "
            f"rust_candle_b64={_f(r_b64.get('load_model_ms')):.3f}  "
            f"rust_tch_seq={_f(r_seq_tch.get('load_model_ms')):.3f}  "
            f"rust_tch_b8={_f(r_b8_tch.get('load_model_ms')):.3f}  "
            f"rust_tch_b64={_f(r_b64_tch.get('load_model_ms')):.3f}  "
            f"python={_f(py.get('load_model_ms')):.3f}"
        )
        return 0

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

    rb8_ms = _f(r_b8.get("total_infer_ms"))
    rb8_sps = _f(r_b8.get("samples_per_sec"))
    print(
        f"{f'Rust batched (bs=8)':30} {rb8_ms:14.3f} {rb8_sps:12.3f}  {r_b8.get('mode', '')}"
    )
    pb8 = _py_batched(py, 8)
    if pb8:
        p_ms = _f(pb8.get("total_infer_ms"))
        p_sps = _f(pb8.get("samples_per_sec"))
        ratio = p_ms / rb8_ms if rb8_ms > 0 else float("inf")
        print(
            f"{f'Python batched (bs=8)':30} {p_ms:14.3f} {p_sps:12.3f}  "
            f"python/rust {ratio:.3f}x"
        )

    rb64_ms = _f(r_b64.get("total_infer_ms"))
    rb64_sps = _f(r_b64.get("samples_per_sec"))
    print(
        f"{f'Rust batched (bs=64)':30} {rb64_ms:14.3f} {rb64_sps:12.3f}  {r_b64.get('mode', '')}"
    )
    pb64 = _py_batched(py, 64)
    if pb64:
        p_ms = _f(pb64.get("total_infer_ms"))
        p_sps = _f(pb64.get("samples_per_sec"))
        ratio = p_ms / rb64_ms if rb64_ms > 0 else float("inf")
        print(
            f"{f'Python batched (bs=64)':30} {p_ms:14.3f} {p_sps:12.3f}  "
            f"python/rust {ratio:.3f}x"
        )

    print()
    print(
        "load_model_ms  "
        f"rust_seq={_f(r_seq.get('load_model_ms')):.3f}  "
        f"rust_b8={_f(r_b8.get('load_model_ms')):.3f}  "
        f"rust_b64={_f(r_b64.get('load_model_ms')):.3f}  "
        f"python={_f(py.get('load_model_ms')):.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
