#!/usr/bin/env python3
"""Run GLiNER2.extract(text, schema) on multitask fixtures; JSON shape matches harness_compare_mt."""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

DEFAULT_MODEL_ID = "fastino/gliner2-base-v1"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixtures", required=True, help="fixtures_multitask.json")
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--out", default="-")
    args = p.parse_args()

    try:
        import torch
    except ImportError:
        print("error: torch required", file=sys.stderr)
        return 1

    device_note = (
        f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    )

    with open(args.fixtures, encoding="utf-8") as f:
        fixtures: list[dict[str, Any]] = json.load(f)

    t0 = time.perf_counter()
    from gliner2 import GLiNER2

    # `from_pretrained` prints a config banner to stdout; keep harness JSON clean.
    _stdout, sys.stdout = sys.stdout, sys.stderr
    try:
        model = GLiNER2.from_pretrained(args.model_id)
    finally:
        sys.stdout = _stdout

    load_model_ms = (time.perf_counter() - t0) * 1000.0

    cases_out: list[dict[str, Any]] = []
    for fix in fixtures:
        fid = fix["id"]
        text = fix["text"]
        schema = fix["schema"]
        threshold = float(fix.get("threshold", 0.5))

        for cls in schema.get("classifications") or []:
            cls.setdefault("true_label", ["N/A"])

        t1 = time.perf_counter()
        result = model.extract(
            text,
            schema,
            threshold=threshold,
            format_results=True,
            include_confidence=False,
            include_spans=False,
        )
        infer_ms = (time.perf_counter() - t1) * 1000.0

        cases_out.append(
            {
                "id": fid,
                "text": text,
                "threshold": threshold,
                "infer_ms": infer_ms,
                "result": result,
            }
        )

    payload = {
        "runner": "python_mt",
        "model_id": args.model_id,
        "device_note": device_note,
        "load_model_ms": load_model_ms,
        "cases": cases_out,
    }

    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out == "-":
        print(text)
    else:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
            f.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
