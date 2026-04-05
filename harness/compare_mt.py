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


def _ms(x: Any) -> float:
    return float(x) if x is not None else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("rust_json", help="harness_compare_mt JSON (Candle)")
    ap.add_argument("python_json")
    ap.add_argument(
        "--rust-tch",
        metavar="PATH",
        help="Optional harness_compare_mt --backend tch JSON",
    )
    ap.add_argument("--tolerance", type=float, default=1e-3)
    ap.add_argument("--warn-only", action="store_true")
    args = ap.parse_args()

    with open(args.rust_json, encoding="utf-8") as f:
        rust = json.load(f)
    with open(args.python_json, encoding="utf-8") as f:
        py = json.load(f)
    rust_tch = None
    if args.rust_tch:
        with open(args.rust_tch, encoding="utf-8") as f:
            rust_tch = json.load(f)

    print("--- metadata ---")
    print(
        f"rust (candle): runner={rust.get('runner')} backend={rust.get('backend', 'candle')} "
        f"model_id={rust.get('model_id')} device={rust.get('device_note')} "
        f"load_ms={_ms(rust.get('load_model_ms')):.3f}"
    )
    if rust_tch:
        print(
            f"rust (tch):    runner={rust_tch.get('runner')} backend={rust_tch.get('backend', 'tch')} "
            f"model_id={rust_tch.get('model_id')} device={rust_tch.get('device_note')} "
            f"load_ms={_ms(rust_tch.get('load_model_ms')):.3f}"
        )
    print(
        f"python: runner={py.get('runner')} model_id={py.get('model_id')} "
        f"device={py.get('device_note')} load_ms={_ms(py.get('load_model_ms')):.3f}"
    )

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

        r_ms = float(rc.get("infer_ms", 0))
        p_ms = float(pc.get("infer_ms", 0))
        ratio = (p_ms / r_ms) if r_ms > 0 else float("inf")
        if rust_tch:
            tc = {c["id"]: c for c in rust_tch.get("cases", [])}.get(cid)
            t_ms = float(tc.get("infer_ms", 0)) if tc else 0.0
            ratio_ct = (t_ms / r_ms) if r_ms > 0 else float("inf")
            ratio_pt = (p_ms / t_ms) if t_ms > 0 else float("inf")
            print(
                f"  timing: rust_candle infer_ms={r_ms:.3f} rust_tch infer_ms={t_ms:.3f} "
                f"python infer_ms={p_ms:.3f} (tch/candle={ratio_ct:.3f}x python/tch={ratio_pt:.3f}x)"
            )
        else:
            print(
                f"  timing: rust infer_ms={r_ms:.3f} python infer_ms={p_ms:.3f} "
                f"(python/rust={ratio:.3f}x)"
            )

    r_total = sum(float(c.get("infer_ms", 0)) for c in rust.get("cases", []))
    p_total = sum(float(c.get("infer_ms", 0)) for c in py.get("cases", []))
    print()
    print("--- total infer_ms (sum of cases) ---")
    if rust_tch:
        t_total = sum(float(c.get("infer_ms", 0)) for c in rust_tch.get("cases", []))
        print(
            f"rust_candle: {r_total:.3f}  rust_tch: {t_total:.3f}  python: {p_total:.3f}  "
            f"python/candle: {(p_total / r_total if r_total else float('inf')):.3f}x  "
            f"tch/candle: {(t_total / r_total if r_total else float('inf')):.3f}x  "
            f"python/tch: {(p_total / t_total if t_total else float('inf')):.3f}x"
        )
    else:
        print(
            f"rust: {r_total:.3f}  python: {p_total:.3f}  "
            f"ratio: {(p_total / r_total if r_total else float('inf')):.3f}x"
        )

    if args.warn_only:
        return 0
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
