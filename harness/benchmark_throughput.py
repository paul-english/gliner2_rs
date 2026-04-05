#!/usr/bin/env python3
"""Throughput benchmark: N entity extractions; sequential and/or batched (PyPI gliner2)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_MODEL_ID = "fastino/gliner2-base-v1"
THROUGHPUT_LABELS = ["company", "person", "product", "location", "date"]


def load_texts(fixtures_path: str, samples: int) -> list[str]:
    with open(fixtures_path, encoding="utf-8") as f:
        base: list[dict[str, Any]] = json.load(f)
    if not base:
        raise SystemExit("fixtures must be non-empty")
    return [str(base[i % len(base)]["text"]) for i in range(samples)]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--fixtures",
        default=None,
        help="fixtures.json (default: harness/fixtures.json next to this script)",
    )
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--warmup", type=int, default=2, help="full passes over all samples")
    p.add_argument("--device", choices=("auto", "cpu"), default="cpu")
    p.add_argument(
        "--mode",
        choices=("sequential", "batched", "both"),
        default="both",
    )
    p.add_argument("--out", default="-")
    args = p.parse_args()

    try:
        import torch
    except ImportError:
        print("error: torch required", file=sys.stderr)
        return 1

    if args.device == "cpu":
        map_location: str | None = "cpu"
        device_note = "cpu"
    else:
        map_location = None
        device_note = (
            f"cuda:{torch.cuda.current_device()}"
            if torch.cuda.is_available()
            else "cpu"
        )

    fixtures_path = args.fixtures or str(
        Path(__file__).resolve().parent / "fixtures.json"
    )
    texts = load_texts(fixtures_path, args.samples)
    threshold = 0.5

    from gliner2 import GLiNER2

    t0 = time.perf_counter()
    _stdout, sys.stdout = sys.stdout, sys.stderr
    try:
        model = GLiNER2.from_pretrained(args.model_id, map_location=map_location)
    finally:
        sys.stdout = _stdout
    load_model_ms = (time.perf_counter() - t0) * 1000.0

    def run_sequential() -> tuple[float, float]:
        """One micro-batch per sample with explicit batch_size=1 (matches Rust sequential)."""
        for _ in range(args.warmup):
            for t in texts:
                model.batch_extract_entities(
                    [t],
                    THROUGHPUT_LABELS,
                    batch_size=1,
                    threshold=threshold,
                    include_confidence=True,
                    include_spans=True,
                )
        t1 = time.perf_counter()
        for t in texts:
            model.batch_extract_entities(
                [t],
                THROUGHPUT_LABELS,
                batch_size=1,
                threshold=threshold,
                include_confidence=True,
                include_spans=True,
            )
        elapsed_ms = (time.perf_counter() - t1) * 1000.0
        sps = args.samples / (elapsed_ms / 1000.0) if elapsed_ms > 0 else float("inf")
        return elapsed_ms, sps

    def run_batched() -> tuple[float, float]:
        bs = args.samples
        for _ in range(args.warmup):
            model.batch_extract_entities(
                texts,
                THROUGHPUT_LABELS,
                batch_size=bs,
                threshold=threshold,
                include_confidence=True,
                include_spans=True,
            )
        t1 = time.perf_counter()
        model.batch_extract_entities(
            texts,
            THROUGHPUT_LABELS,
            batch_size=bs,
            threshold=threshold,
            include_confidence=True,
            include_spans=True,
        )
        elapsed_ms = (time.perf_counter() - t1) * 1000.0
        sps = args.samples / (elapsed_ms / 1000.0) if elapsed_ms > 0 else float("inf")
        return elapsed_ms, sps

    payload: dict[str, Any] = {
        "runner": "python_throughput",
        "model_id": args.model_id,
        "device_note": device_note,
        "fixtures": fixtures_path,
        "samples": args.samples,
        "warmup_full_passes": args.warmup,
        "load_model_ms": load_model_ms,
    }

    if args.mode in ("sequential", "both"):
        total_ms, sps = run_sequential()
        payload["sequential"] = {
            "mode": "sequential",
            "batch_size": 1,
            "total_infer_ms": total_ms,
            "samples_per_sec": sps,
        }

    if args.mode in ("batched", "both"):
        total_ms, sps = run_batched()
        payload["batched"] = {
            "mode": "batched",
            "batch_size": args.samples,
            "total_infer_ms": total_ms,
            "samples_per_sec": sps,
        }

    text = json.dumps(payload, indent=2)
    if args.out == "-":
        print(text)
    else:
        Path(args.out).write_text(text + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
