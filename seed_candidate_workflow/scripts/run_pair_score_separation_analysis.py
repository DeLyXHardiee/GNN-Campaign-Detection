"""CLI entrypoint for pair score separation analysis (implementation in utils)."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from seed_candidate_workflow.utils.pair_score_separation import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
