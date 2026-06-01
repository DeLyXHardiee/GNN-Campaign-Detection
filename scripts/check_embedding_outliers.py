"""
Check SBERT embeddings for NaN/Inf values that would corrupt normalization.
Reports per-field counts and lists affected entry keys.
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

EMBEDDINGS_PATH = Path(__file__).parent.parent / "core/utils/embeddings/output/embeddings.json"
FIELDS = ("subj", "body")


def check_vector(vec: list) -> tuple[int, int]:
    """Return (nan_count, inf_count) for a float vector."""
    nan_count = sum(1 for v in vec if math.isnan(v))
    inf_count = sum(1 for v in vec if math.isinf(v))
    return nan_count, inf_count


def main() -> None:
    print(f"Loading {EMBEDDINGS_PATH} ...")
    with open(EMBEDDINGS_PATH) as f:
        data = json.load(f)

    by_key: dict = data["by_key"]
    total = len(by_key)
    print(f"Entries: {total}\n")

    bad: dict[str, list] = defaultdict(list)

    for entry_key, entry in by_key.items():
        for field in FIELDS:
            vec = entry.get(field)
            if vec is None:
                bad[field].append((entry_key, "MISSING", 0))
                continue
            nans, infs = check_vector(vec)
            if nans or infs:
                bad[field].append((entry_key, nans, infs))

    any_bad = any(bad[f] for f in FIELDS)

    if not any_bad:
        print("No NaN or Inf values found. Embeddings look clean.")
        sys.exit(0)

    print("Outliers detected:\n")
    for field in FIELDS:
        issues = bad[field]
        if not issues:
            print(f"  {field}: OK")
            continue

        total_nan = sum(n for _, n, _ in issues if isinstance(n, int))
        total_inf = sum(i for _, _, i in issues if isinstance(i, int))
        print(f"  {field}: {len(issues)} affected entries "
              f"({total_nan} NaN values, {total_inf} Inf values)")
        for entry_key, nans, infs in issues:
            print(f"    key={entry_key[:60]}...  nans={nans}  infs={infs}")

    print(f"\nSummary: {sum(len(bad[f]) for f in FIELDS)} total affected "
          f"entries across {len([f for f in FIELDS if bad[f]])} field(s).")
    sys.exit(1)


if __name__ == "__main__":
    main()
