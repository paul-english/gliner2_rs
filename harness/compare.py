#!/usr/bin/env python3
"""Compare two harness JSON outputs (Rust vs Python): entities, confidence, timing."""

from __future__ import annotations

import argparse
import json
from typing import Any


def entity_key(e: dict[str, Any]) -> tuple[str, int, int]:
    return (str(e["label"]), int(e["start"]), int(e["end"]))


def load_harness(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compare_case(
    rust_case: dict[str, Any],
    py_case: dict[str, Any],
    conf_tolerance: float,
) -> tuple[list[str], bool]:
    """Return (lines, ok)."""
    lines: list[str] = []
    ok = True
    cid = rust_case["id"]
    lines.append(f"=== case {cid!r} ===")

    r_entities = {entity_key(e): e for e in rust_case.get("entities", [])}
    p_entities = {entity_key(e): e for e in py_case.get("entities", [])}

    r_keys = set(r_entities)
    p_keys = set(p_entities)

    only_rust = sorted(r_keys - p_keys)
    only_py = sorted(p_keys - r_keys)
    if only_rust:
        ok = False
        lines.append(f"  only in rust ({len(only_rust)}): {only_rust}")
    if only_py:
        ok = False
        lines.append(f"  only in python ({len(only_py)}): {only_py}")

    for key in sorted(r_keys & p_keys):
        r, p = r_entities[key], p_entities[key]
        rt = str(r.get("text", "")).strip()
        pt = str(p.get("text", "")).strip()
        if rt != pt:
            ok = False
            lines.append(f"  text mismatch {key}: rust={rt!r} python={pt!r}")

        rc = float(r["confidence"])
        pc = float(p["confidence"])
        diff = abs(rc - pc)
        if diff > conf_tolerance:
            ok = False
            lines.append(
                f"  confidence mismatch {key}: rust={rc:.6f} python={pc:.6f} (|Δ|={diff:.6g})"
            )
        elif diff > 0:
            lines.append(
                f"  confidence (within tol) {key}: rust={rc:.6f} python={pc:.6f} (|Δ|={diff:.6g})"
            )

    r_ms = float(rust_case.get("infer_ms", 0))
    p_ms = float(py_case.get("infer_ms", 0))
    ratio = (p_ms / r_ms) if r_ms > 0 else float("inf")
    lines.append(
        f"  timing: rust infer_ms={r_ms:.3f} python infer_ms={p_ms:.3f} (python/rust={ratio:.3f}x)"
    )

    if ok and not only_rust and not only_py:
        lines.append("  entities: OK (spans + text after strip + confidence)")

    return lines, ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("rust_json", help="JSON from harness_compare")
    ap.add_argument("python_json", help="JSON from run_python.py")
    ap.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
        help="Max allowed |Δconfidence| per span (default: 1e-3)",
    )
    ap.add_argument(
        "--warn-only",
        action="store_true",
        help="Print mismatches but always exit 0",
    )
    args = ap.parse_args()

    rust = load_harness(args.rust_json)
    py = load_harness(args.python_json)

    def _ms(x: Any) -> float:
        return float(x) if x is not None else 0.0

    print("--- metadata ---")
    print(
        f"rust:   runner={rust.get('runner')} model_id={rust.get('model_id')} "
        f"device={rust.get('device_note')} load_ms={_ms(rust.get('load_model_ms')):.3f}"
    )
    print(
        f"python: runner={py.get('runner')} model_id={py.get('model_id')} "
        f"device={py.get('device_note')} load_ms={_ms(py.get('load_model_ms')):.3f}"
    )

    r_cases = {c["id"]: c for c in rust.get("cases", [])}
    p_cases = {c["id"]: c for c in py.get("cases", [])}
    ids = sorted(set(r_cases) | set(p_cases))

    all_ok = True
    for cid in ids:
        if cid not in r_cases:
            print(f"\n=== case {cid!r} MISSING in rust output ===")
            all_ok = False
            continue
        if cid not in p_cases:
            print(f"\n=== case {cid!r} MISSING in python output ===")
            all_ok = False
            continue
        lines, case_ok = compare_case(r_cases[cid], p_cases[cid], args.tolerance)
        print()
        print("\n".join(lines))
        if not case_ok:
            all_ok = False

    r_total = sum(float(c.get("infer_ms", 0)) for c in rust.get("cases", []))
    p_total = sum(float(c.get("infer_ms", 0)) for c in py.get("cases", []))
    print()
    print("--- total infer_ms (sum of cases) ---")
    print(
        f"rust: {r_total:.3f}  python: {p_total:.3f}  ratio: {(p_total / r_total if r_total else float('inf')):.3f}x"
    )

    if args.warn_only:
        return 0
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
