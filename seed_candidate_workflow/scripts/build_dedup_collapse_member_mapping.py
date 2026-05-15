#!/usr/bin/env python3
"""Write semantic-supernode-compatible member mapping JSON from MISP dedup collapse sidecars."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from seed_candidate_workflow.utils import semantic_supernode_gt_metrics as mgt  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dedup-collapse-out-dir",
        type=Path,
        default=REPO / "data" / "misp" / "misp_lake_dedup_task_identity",
        help="Directory with external_id_map.csv from collapse_misp_lake_strict_duplicates.py",
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=REPO
        / "data"
        / "misp"
        / "misp_lake_dedup_task_identity"
        / "dedup_task_identity_member_mapping.json",
    )
    args = ap.parse_args()
    out_dir = args.dedup_collapse_out_dir.expanduser().resolve()
    tab = mgt.load_dedup_collapse_member_table_from_out_dir(out_dir)
    summ_path = out_dir / "collapse_summary.json"
    meta: dict[str, object] = {"dedup_collapse_out_dir": str(out_dir)}
    if summ_path.is_file():
        summ = json.loads(summ_path.read_text(encoding="utf-8"))
        meta["collapse_signature_type"] = summ.get("collapse_signature_type")
        meta["n_events_in"] = summ.get("n_events_in")
        meta["n_events_out"] = summ.get("n_events_out")
    out_path = args.out_json.expanduser().resolve()
    mgt.write_member_expansion_mapping_json(out_path, gid_to_members=tab, meta=meta)
    print(f"Wrote {len(tab)} representative nodes -> {out_path}")


if __name__ == "__main__":
    main()
