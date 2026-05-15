#!/usr/bin/env python3
"""One-off repair: stray digit after comma before \"value\" (invalid JSON)."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    path = args.json_path.expanduser().resolve()
    text = path.read_text(encoding="utf-8")
    bad = '"type": "header_Received",2'
    if bad not in text:
        print("pattern not found; nothing to fix")
        return 0
    fixed = text.replace(bad, '"type": "header_Received",', 1)
    if fixed == text:
        print("replace had no effect")
        return 1
    try:
        json.loads(fixed)
    except json.JSONDecodeError as e:
        print("still invalid after replace:", e)
        return 1
    if args.dry_run:
        print("dry-run OK: would fix one occurrence")
        return 0
    bak = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, bak)
    path.write_text(fixed, encoding="utf-8")
    print(f"fixed {path} (backup {bak})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
