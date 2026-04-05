#!/usr/bin/env python3
"""Print sorted tensor keys in a safetensors file (developer utility)."""

from __future__ import annotations

import sys

from safetensors import safe_open


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/list_keys.py <safetensors_file>", file=sys.stderr)
        sys.exit(1)
    path = sys.argv[1]
    with safe_open(path, framework="pt") as f:
        for key in sorted(f.keys()):
            print(key)


if __name__ == "__main__":
    main()
