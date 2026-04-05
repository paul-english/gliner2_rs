#!/usr/bin/env python3
"""Run PyPI gliner2 on harness fixtures; emit JSON matching the Rust harness_compare schema."""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

DEFAULT_MODEL_ID = "fastino/gliner2-base-v1"


def flatten_entities(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Turn result['entities'] label -> list of span dicts into a flat sorted list."""
    out: list[dict[str, Any]] = []
    entities = result.get("entities") or {}
    if not isinstance(entities, dict):
        return out
    for label, spans in entities.items():
        if spans is None:
            continue
        if not isinstance(spans, list):
            continue
        for item in spans:
            if isinstance(item, str):
                # No spans/confidence in this mode — skip detailed compare
                continue
            if not isinstance(item, dict):
                continue
            text = item.get("text", "")
            out.append(
                {
                    "label": label,
                    "text": text,
                    "start": int(item["start"]),
                    "end": int(item["end"]),
                    "confidence": float(item["confidence"]),
                }
            )
    out.sort(key=lambda e: (e["label"], e["start"], e["end"]))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--fixtures",
        required=True,
        help="Path to fixtures.json (same format as Rust harness)",
    )
    p.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"Hugging Face model id (default: {DEFAULT_MODEL_ID})",
    )
    p.add_argument(
        "--out",
        default="-",
        help="Output path, or '-' for stdout (default: -)",
    )
    p.add_argument(
        "--device",
        choices=("auto", "cpu"),
        default="auto",
        help="auto: use CUDA when available; cpu: load and run on CPU (harness CPU-vs-CPU)",
    )
    args = p.parse_args()

    try:
        import torch
    except ImportError:
        print("error: torch is required (pulled in by gliner2)", file=sys.stderr)
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

    with open(args.fixtures, encoding="utf-8") as f:
        fixtures: list[dict[str, Any]] = json.load(f)

    t0 = time.perf_counter()
    from gliner2 import GLiNER2

    _stdout, sys.stdout = sys.stdout, sys.stderr
    try:
        model = GLiNER2.from_pretrained(args.model_id, map_location=map_location)
    finally:
        sys.stdout = _stdout

    load_model_ms = (time.perf_counter() - t0) * 1000.0

    cases_out: list[dict[str, Any]] = []
    for fix in fixtures:
        fid = fix["id"]
        text = fix["text"]
        entity_types = fix["entity_types"]
        threshold = float(fix["threshold"])

        t1 = time.perf_counter()
        # Explicit batch_size=1 (same as Rust harness single-forward path).
        result = model.batch_extract_entities(
            [text],
            list(entity_types),
            batch_size=1,
            threshold=threshold,
            include_confidence=True,
            include_spans=True,
        )[0]
        infer_ms = (time.perf_counter() - t1) * 1000.0

        cases_out.append(
            {
                "id": fid,
                "text": text,
                "entity_types": list(entity_types),
                "threshold": threshold,
                "infer_ms": infer_ms,
                "entities": flatten_entities(result),
            }
        )

    payload = {
        "runner": "python",
        "model_id": args.model_id,
        "device_note": device_note,
        "load_model_ms": load_model_ms,
        "cases": cases_out,
    }

    text = json.dumps(payload, indent=2)
    if args.out == "-":
        print(text)
    else:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
            f.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
