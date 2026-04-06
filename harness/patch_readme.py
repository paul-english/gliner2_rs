#!/usr/bin/env python3
"""Replace gliner2-harness marker regions in README.md with generated markdown."""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
from datetime import date
from pathlib import Path

# Ensure sibling harness modules import when the script is run as e.g. `python harness/patch_readme.py`
# from the repo root (sys.path[0] is not always the harness dir).
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from readme_tables import render_all_fragments, resolve_artifact_dir, tch_json_for

# Must exist in README when patching from a full artifact dir (entity + multitask).
REQUIRED_MARKERS = frozenset(
    {
        "cpu-recorded",
        "entity-summary",
        "entity-footnote",
        "entity-cases",
        "multitask-summary",
    }
)

MARKER_START = "<!-- gliner2-harness:{name} -->"
MARKER_END = "<!-- /gliner2-harness:{name} -->"


def patch_region(text: str, name: str, inner: str) -> tuple[str, bool]:
    start = MARKER_START.format(name=name)
    end = MARKER_END.format(name=name)
    pattern = re.compile(
        re.escape(start) + r"\n?([\s\S]*?)\n?" + re.escape(end),
        re.MULTILINE,
    )
    if not pattern.search(text):
        return text, False
    block = f"{start}\n{inner.rstrip()}\n{end}"
    # Callable replacement avoids re.sub interpreting backslashes in markdown (e.g. \d).
    new_text, n = pattern.subn(lambda _m: block, text, count=1)
    return new_text, n > 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--readme",
        type=Path,
        required=True,
        help="Path to README.md (repo root)",
    )
    ap.add_argument(
        "--artifact-dir",
        type=Path,
        required=True,
        help="Directory with JSON (e.g. harness/.compare_last). Relative paths resolve from the README repo root.",
    )
    ap.add_argument(
        "--date",
        default="",
        help="Recorded date YYYY-MM-DD (default: today)",
    )
    ap.add_argument(
        "--platform-note",
        default="",
        help="Platform string for Recorded lines (default: uname)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print fragments to stdout; do not write README",
    )
    args = ap.parse_args()

    recorded_date = args.date or date.today().isoformat()
    plat = args.platform_note or (
        f"{platform.system()} {platform.machine()}, local run"
    )

    readme_path = args.readme.expanduser().resolve()
    artifact_dir = resolve_artifact_dir(args.artifact_dir, readme_path)

    try:
        fragments = render_all_fragments(artifact_dir, recorded_date, plat)
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    entity_candle = artifact_dir / "entity_rust.json"
    mt_candle = artifact_dir / "mt_rust.json"
    for label, candle in ("entity", entity_candle), ("multitask", mt_candle):
        tch_path = tch_json_for(candle)
        if candle.is_file() and not tch_path.is_file():
            print(
                f"warning: missing {tch_path.name} ({label} tch-rs); "
                "README tch-rs columns will show em dashes. "
                "Run with GLINER2_BENCH_TCH=1 or bash harness/run_compare_all.sh (default).",
                file=sys.stderr,
            )

    if args.dry_run:
        for k, v in fragments.items():
            print(f"--- {k} ---\n{v}\n")
        return 0

    readme_text = readme_path.read_text(encoding="utf-8")
    missing: list[str] = []
    missing_required: list[str] = []
    for name, inner in fragments.items():
        readme_text, ok = patch_region(readme_text, name, inner)
        if not ok:
            missing.append(name)
            if name in REQUIRED_MARKERS:
                missing_required.append(name)

    if missing:
        print(
            f"warning: marker(s) not found in {readme_path} (skipped): {missing}",
            file=sys.stderr,
        )
    if missing_required:
        print(
            f"error: required marker(s) missing in {readme_path}: {missing_required}",
            file=sys.stderr,
        )
        return 1

    readme_path.write_text(readme_text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
