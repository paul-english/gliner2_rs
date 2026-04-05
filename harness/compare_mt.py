#!/usr/bin/env python3
"""Compare multitask harness JSON (Rust vs Python): deep equality with float tolerance."""

from __future__ import annotations

import argparse
import json
import math
from typing import Any


def approx_equal(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol or (math.isnan(a) and math.isnan(b))


def deep_compare(
    a: Any,
    b: Any,
    tol: float,
    path: str,
) -> tuple[list[str], bool]:
    lines: list[str] = []
    ok = True
    if type(a) is not type(b):
        lines.append(f"{path}: type {type(a).__name__} != {type(b).__name__}")
        return lines, False
    if isinstance(a, dict):
        ak, bk = set(a), set(b)
        if ak != bk:
            ok = False
            lines.append(
                f"{path}: keys only rust={sorted(ak - bk)} only py={sorted(bk - ak)}"
            )
        for k in sorted(ak & bk):
            sl, sk = deep_compare(a[k], b[k], tol, f"{path}.{k}")
            lines.extend(sl)
            ok = ok and sk
        return lines, ok
    if isinstance(a, list):
        if len(a) != len(b):
            lines.append(f"{path}: len {len(a)} != {len(b)}")
            return lines, False
        for i, (x, y) in enumerate(zip(a, b)):
            sl, sk = deep_compare(x, y, tol, f"{path}[{i}]")
            lines.extend(sl)
            ok = ok and sk
        return lines, ok
    if isinstance(a, float) and isinstance(b, float):
        if not approx_equal(a, b, tol):
            lines.append(f"{path}: {a!r} != {b!r} (tol={tol})")
            return lines, False
        return lines, True
    if a != b:
        lines.append(f"{path}: {a!r} != {b!r}")
        return lines, False
    return lines, True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("rust_json")
    ap.add_argument("python_json")
    ap.add_argument("--tolerance", type=float, default=1e-3)
    ap.add_argument("--warn-only", action="store_true")
    args = ap.parse_args()

    with open(args.rust_json, encoding="utf-8") as f:
        rust = json.load(f)
    with open(args.python_json, encoding="utf-8") as f:
        py = json.load(f)

    print("--- metadata ---")
    print(f"rust:   {rust.get('runner')} {rust.get('model_id')}")
    print(f"python: {py.get('runner')} {py.get('model_id')}")

    r_cases = {c["id"]: c for c in rust.get("cases", [])}
    p_cases = {c["id"]: c for c in py.get("cases", [])}
    all_ok = True
    for cid in sorted(set(r_cases) | set(p_cases)):
        if cid not in r_cases or cid not in p_cases:
            print(f"\nMISSING case {cid!r}")
            all_ok = False
            continue
        rc, pc = r_cases[cid], p_cases[cid]
        print(f"\n=== case {cid!r} ===")
        sl, sk = deep_compare(
            rc.get("result"), pc.get("result"), args.tolerance, "result"
        )
        for line in sl:
            print(f"  {line}")
        if sk and not sl:
            print("  result: OK")
        all_ok = all_ok and sk

    if args.warn_only:
        return 0
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
