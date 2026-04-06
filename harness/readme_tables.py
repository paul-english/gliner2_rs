"""Build markdown fragments for README comparison tables from harness JSON artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def _is_close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-6)


def _bold_if_min_ms(values: list[float], ndigits: int = 1) -> list[str]:
    """Format ms values; bold those equal to the minimum (ties all bold)."""
    if not values:
        return []
    best = min(values)
    out: list[str] = []
    for v in values:
        s = f"{v:.{ndigits}f}"
        out.append(f"**{s}**" if _is_close(v, best) else s)
    return out


def _bold_if_max_sps(values: list[float], ndigits: int = 2) -> list[str]:
    """Format samples/s; bold those equal to the maximum (ties all bold)."""
    if not values:
        return []
    best = max(values)
    out: list[str] = []
    for v in values:
        s = f"{v:.{ndigits}f}"
        out.append(f"**{s}**" if _is_close(v, best) else s)
    return out


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _f(x: Any) -> float:
    return float(x) if x is not None else 0.0


def _ms_line(cases: list[dict[str, Any]]) -> float:
    return sum(_f(c.get("infer_ms")) for c in cases)


def render_cpu_recorded_paragraph(recorded_date: str, platform_note: str) -> str:
    return (
        f"Model: `fastino/gliner2-base-v1`. **Recorded:** {recorded_date} ({platform_note}; "
        "numbers vary by machine and load). **tch-rs `infer_ms`:** LibTorch encoder path with "
        "`download-libtorch` + [prepend_libtorch_ld_path.sh](harness/prepend_libtorch_ld_path.sh); "
        "see caveat above on NER outputs vs Candle."
    )


def render_entity_summary(
    rust: dict[str, Any],
    py: dict[str, Any],
    rust_tch: dict[str, Any] | None,
) -> str:
    r_cases = rust.get("cases", [])
    r_total = _ms_line(r_cases)
    p_total = _ms_line(py.get("cases", []))

    r_load = _f(rust.get("load_model_ms"))
    p_load = _f(py.get("load_model_ms"))

    if rust_tch:
        t_load = _f(rust_tch.get("load_model_ms"))
        br, bt, bp = _bold_if_min_ms([r_load, t_load, p_load])
        t_sum = _ms_line(rust_tch.get("cases", []))
        sr, st, sp = _bold_if_min_ms([r_total, t_sum, p_total])
        row_load = (
            f"| `load_model_ms`              | {br}         | "
            f"{bt}          | {bp}          |"
        )
        row_sum = (
            f"| Sum of `infer_ms` over cases | {sr}         | "
            f"{st}           | {sp}            |"
        )
    else:
        br, bp = _bold_if_min_ms([r_load, p_load])
        sr, sp = _bold_if_min_ms([r_total, p_total])
        row_load = (
            f"| `load_model_ms`              | {br}         | "
            f"—               | {bp}          |"
        )
        row_sum = (
            f"| Sum of `infer_ms` over cases | {sr}         | "
            f"—               | {sp}            |"
        )

    lines = [
        "|                              | Rust (Candle) | Rust (tch-rs)   | Python           |",
        "| ---------------------------- | ------------- | --------------- | ---------------- |",
        f"| `device_note`                | `{rust.get('device_note', '')}`         | "
        f"`{rust_tch.get('device_note', '—') if rust_tch else '—'}` | `"
        f"{py.get('device_note', '')}`            |",
        row_load,
        row_sum,
    ]

    if rust_tch:
        t_total = _ms_line(rust_tch.get("cases", []))
        tch_cnd = t_total / r_total if r_total else float("inf")
        py_cnd = p_total / r_total if r_total else float("inf")
        py_tch = p_total / t_total if t_total else float("inf")
        lines.append(
            "| Ratios (total infer)         | —             | "
            f"tch/cnd {tch_cnd:.2f}× | "
            f"py/cnd {py_cnd:.2f}×; py/tch {py_tch:.2f}× |"
        )
    else:
        py_cnd = p_total / r_total if r_total else float("inf")
        lines.append(
            "| Ratios (total infer)         | —             | —               | "
            f"py/cnd {py_cnd:.2f}× |"
        )

    return "\n".join(lines)


def render_entity_cases(
    rust: dict[str, Any],
    py: dict[str, Any],
    rust_tch: dict[str, Any] | None,
) -> str:
    r_cases = {c["id"]: c for c in rust.get("cases", [])}
    p_cases = {c["id"]: c for c in py.get("cases", [])}
    t_cases = {c["id"]: c for c in rust_tch.get("cases", [])} if rust_tch else None
    ids = sorted(set(r_cases) | set(p_cases))

    lines = [
        "| Case id             | Candle `infer_ms` | tch-rs `infer_ms` | python `infer_ms` | `python/candle` | `python/tch` |",
        "| ------------------- | ----------------- | ----------------- | ----------------- | --------------- | ------------ |",
    ]
    for cid in ids:
        if cid not in r_cases or cid not in p_cases:
            continue
        r_ms = _f(r_cases[cid].get("infer_ms"))
        p_ms = _f(p_cases[cid].get("infer_ms"))
        ratio_pc = p_ms / r_ms if r_ms else float("inf")
        t_ms = _f(t_cases[cid].get("infer_ms")) if t_cases and cid in t_cases else None
        if t_ms is not None:
            fr, ft, fp = _bold_if_min_ms([r_ms, t_ms, p_ms])
            t_cell = ft
        else:
            fr, fp = _bold_if_min_ms([r_ms, p_ms])
            t_cell = "—"
        if t_ms is not None and t_ms > 0:
            ratio_pt = p_ms / t_ms
            pt_cell = f"{ratio_pt:.2f}×"
        else:
            pt_cell = "—"
        lines.append(
            f"| `{cid}` | {fr} | {t_cell} | {fp} | {ratio_pc:.2f}× | {pt_cell} |"
        )
    return "\n".join(lines)


def render_entity_footnote() -> str:
    return "† Expected device label for tch-rs harness JSON when LibTorch is used (`run_compare_all.sh` enables this by default; otherwise set `GLINER2_BENCH_TCH=1`)."


def render_multitask_summary(
    rust: dict[str, Any],
    py: dict[str, Any],
    rust_tch: dict[str, Any] | None,
) -> str:
    r_cases = rust.get("cases", [])
    p_cases = py.get("cases", [])
    r_total = _ms_line(r_cases)
    p_total = _ms_line(p_cases)
    t_total = _ms_line(rust_tch.get("cases", [])) if rust_tch else None

    r_load = _f(rust.get("load_model_ms"))
    p_load = _f(py.get("load_model_ms"))

    row_device = (
        f"| `device_note`        | `{rust.get('device_note', '')}`         | "
        f"`{rust_tch.get('device_note', '—') if rust_tch else '—'}` | `"
        f"{py.get('device_note', '')}`            |"
    )
    if rust_tch:
        t_load = _f(rust_tch.get("load_model_ms"))
        br, bt, bp = _bold_if_min_ms([r_load, t_load, p_load])
        tt = t_total if t_total is not None else 0.0
        sr, st, sp = _bold_if_min_ms([r_total, tt, p_total])
        row_load = (
            f"| `load_model_ms`      | {br}         | {bt}          | {bp}           |"
        )
        row_sum = (
            f"| Sum of `infer_ms`    | {sr}         | "
            f"{st}           | {sp}            |"
        )
    else:
        br, bp = _bold_if_min_ms([r_load, p_load])
        sr, sp = _bold_if_min_ms([r_total, p_total])
        row_load = (
            f"| `load_model_ms`      | {br}         | "
            f"—               | {bp}           |"
        )
        row_sum = (
            f"| Sum of `infer_ms`    | {sr}         | "
            f"—               | {sp}            |"
        )

    lines = [
        "|                      | Rust (Candle) | Rust (tch-rs)   | Python           |",
        "| -------------------- | ------------- | --------------- | ---------------- |",
        row_device,
        row_load,
        row_sum,
    ]

    if rust_tch and t_total is not None:
        tch_cnd = t_total / r_total if r_total else float("inf")
        py_cnd = p_total / r_total if r_total else float("inf")
        py_tch = p_total / t_total if t_total else float("inf")
        lines.append(
            "| Ratios (total infer) | —             | "
            f"tch/cnd {tch_cnd:.2f}× | "
            f"py/cnd {py_cnd:.2f}×; py/tch {py_tch:.2f}× |"
        )
    else:
        py_cnd = p_total / r_total if r_total else float("inf")
        lines.append(
            "| Ratios (total infer) | —             | —               | "
            f"py/cnd {py_cnd:.2f}× |"
        )

    return "\n".join(lines)


def _py_batched(py: dict[str, Any], bs: int) -> dict[str, Any] | None:
    by = py.get("batched_by_size")
    if isinstance(by, dict) and str(bs) in by:
        return by[str(bs)]
    b = py.get("batched")
    if b is not None and int(b.get("batch_size", -1)) == bs:
        return b
    return None


def render_throughput_recorded(recorded_date: str, platform_note: str) -> str:
    return (
        f"**Recorded:** {recorded_date} ({platform_note}, CPU, `CUDA_VISIBLE_DEVICES=` + "
        "`--device cpu` on Python). `warmup_full_passes=8` over all samples before each timed pass. "
        "[harness/compare_throughput.py](harness/compare_throughput.py) prints Candle vs tch vs Python "
        "(ratios: `py/cnd`, `tch/cnd`, `py/tch`)."
    )


def render_throughput_table(
    r_seq: dict[str, Any],
    r_b8: dict[str, Any],
    r_b64: dict[str, Any],
    py: dict[str, Any],
    r_seq_t: dict[str, Any] | None,
    r_b8_t: dict[str, Any] | None,
    r_b64_t: dict[str, Any] | None,
) -> str:
    """README-style wide table (tch columns omitted or filled with —)."""

    def row(
        label: str,
        rs: dict[str, Any],
        rt: dict[str, Any] | None,
        py_bs: int,
    ) -> str:
        rs_ms = _f(rs.get("total_infer_ms"))
        rs_sps = _f(rs.get("samples_per_sec"))
        if py_bs == 1:
            ps = py.get("sequential")
            p_ms = _f(ps.get("total_infer_ms")) if ps else 0.0
            p_sps = _f(ps.get("samples_per_sec")) if ps else 0.0
        else:
            pb = _py_batched(py, py_bs)
            p_ms = _f(pb.get("total_infer_ms")) if pb else 0.0
            p_sps = _f(pb.get("samples_per_sec")) if pb else 0.0
        if rt is not None:
            rt_ms = _f(rt.get("total_infer_ms"))
            rt_sps = _f(rt.get("samples_per_sec"))
            py_cnd = p_ms / rs_ms if rs_ms else float("inf")
            py_tch = p_ms / rt_ms if rt_ms else float("inf")
            ms_b = _bold_if_min_ms([rs_ms, rt_ms, p_ms], ndigits=0)
            sps_b = _bold_if_max_sps([rs_sps, rt_sps, p_sps], ndigits=2)
            return (
                f"| {label:30} | {ms_b[0]} | {sps_b[0]} | {ms_b[1]} | "
                f"{sps_b[1]} | {ms_b[2]} | {sps_b[2]} | "
                f"{py_cnd:.2f}× | {py_tch:.2f}× |"
            )
        py_cnd = p_ms / rs_ms if rs_ms else float("inf")
        ms_b = _bold_if_min_ms([rs_ms, p_ms], ndigits=0)
        sps_b = _bold_if_max_sps([rs_sps, p_sps], ndigits=2)
        return (
            f"| {label:30} | {ms_b[0]} | {sps_b[0]} | — | — | "
            f"{ms_b[1]} | {sps_b[1]} | {py_cnd:.2f}× | — |"
        )

    header = (
        "| Lane                           | Candle `infer_ms` | Candle s/s | "
        "tch-rs `infer_ms` | tch-rs s/s | Python `infer_ms` | Python s/s | py/candle | py/tch  |"
    )
    sep = (
        "| ------------------------------ | ----------------- | ---------- | "
        "----------------- | ---------- | ----------------- | ---------- | --------- | ------- |"
    )
    lines = [
        header,
        sep,
        row("Sequential (`batch_size` 1)", r_seq, r_seq_t, 1),
        row("Batched (`batch_size` 8)", r_b8, r_b8_t, 8),
        row("Batched (`batch_size` 64)", r_b64, r_b64_t, 64),
    ]
    return "\n".join(lines)


def render_throughput_loads(
    r_seq: dict[str, Any],
    r_b8: dict[str, Any],
    r_b64: dict[str, Any],
    py: dict[str, Any],
    r_seq_t: dict[str, Any] | None,
    r_b8_t: dict[str, Any] | None,
    r_b64_t: dict[str, Any] | None,
) -> str:
    """Single-line load summary; uses seq run load for each Rust backend."""
    c_ms = _f(r_seq.get("load_model_ms"))
    py_ms = _f(py.get("load_model_ms"))
    if r_seq_t is not None:
        t_ms = _f(r_seq_t.get("load_model_ms"))
        bc, bt, bp = _bold_if_min_ms([c_ms, t_ms, py_ms], ndigits=0)
        return f"Load times: Candle ~{bc} ms; tch ~{bt} ms; Python ~{bp} ms."
    bc, bp = _bold_if_min_ms([c_ms, py_ms], ndigits=0)
    return f"Load times: Candle ~{bc} ms (seq/b8/b64 runs may differ slightly); Python ~{bp} ms."


def optional_tch(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return _load(path)


def tch_json_for(candle_json: Path) -> Path:
    """Mirror run_all.sh: strip .json from the candle path and add _tch.json (same directory)."""
    return candle_json.with_name(f"{candle_json.stem}_tch.json")


def resolve_artifact_dir(artifact_dir: Path, readme: Path) -> Path:
    """Resolve artifact dir relative to the repo root (README parent), not only the process cwd.

    Relative ``.compare_last`` from the repo root would wrongly resolve to ``<repo>/.compare_last``;
    the harness writes to ``harness/.compare_last`` instead.
    """
    ad = artifact_dir.expanduser()
    if ad.is_absolute():
        return ad.resolve()
    repo_root = readme.parent.resolve()
    cand = (repo_root / ad).resolve()
    if cand.is_dir():
        return cand
    if ad.name == ".compare_last":
        alt = repo_root / "harness" / ".compare_last"
        if alt.is_dir():
            return alt.resolve()
    return cand


def render_all_fragments(
    artifact_dir: Path,
    recorded_date: str,
    platform_note: str,
) -> dict[str, str]:
    """Return marker name -> inner markdown (no surrounding markers)."""
    entity_candle = artifact_dir / "entity_rust.json"
    entity_rust = _load(entity_candle)
    entity_py = _load(artifact_dir / "entity_python.json")
    entity_tch = optional_tch(tch_json_for(entity_candle))

    mt_candle = artifact_dir / "mt_rust.json"
    mt_rust = _load(mt_candle)
    mt_py = _load(artifact_dir / "mt_python.json")
    mt_tch = optional_tch(tch_json_for(mt_candle))

    out: dict[str, str] = {
        "cpu-recorded": render_cpu_recorded_paragraph(recorded_date, platform_note),
        "entity-summary": render_entity_summary(entity_rust, entity_py, entity_tch),
        "entity-footnote": render_entity_footnote(),
        "entity-cases": render_entity_cases(entity_rust, entity_py, entity_tch),
        "multitask-summary": render_multitask_summary(mt_rust, mt_py, mt_tch),
    }

    tp_seq = artifact_dir / "throughput_rust_seq_candle.json"
    if tp_seq.is_file():
        r_seq = _load(tp_seq)
        r_b8 = _load(artifact_dir / "throughput_rust_batch_8_candle.json")
        r_b64 = _load(artifact_dir / "throughput_rust_batch_64_candle.json")
        tp_py = _load(artifact_dir / "throughput_python.json")
        # run_throughput.sh names tch files: <candle_path_stem>_tch.json
        r_seq_t = optional_tch(artifact_dir / "throughput_rust_seq_candle_tch.json")
        r_b8_t = optional_tch(artifact_dir / "throughput_rust_batch_8_candle_tch.json")
        r_b64_t = optional_tch(
            artifact_dir / "throughput_rust_batch_64_candle_tch.json"
        )
        out["throughput-recorded"] = render_throughput_recorded(
            recorded_date, platform_note
        )
        out["throughput-table"] = render_throughput_table(
            r_seq, r_b8, r_b64, tp_py, r_seq_t, r_b8_t, r_b64_t
        )
        out["throughput-loads"] = render_throughput_loads(
            r_seq, r_b8, r_b64, tp_py, r_seq_t, r_b8_t, r_b64_t
        )
    return out
