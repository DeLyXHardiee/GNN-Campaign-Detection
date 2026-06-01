"""
Run the embedding component independently: load-or-compute embeddings for all
emails from a MISP JSON file, using the module's output folder.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    _core = Path(__file__).resolve().parent.parent.parent
    if str(_core) not in sys.path:
        sys.path.insert(0, str(_core))
    from utils.embeddings import DEFAULT_OUTPUT_DIR, run_standalone
else:
    from . import DEFAULT_OUTPUT_DIR, run_standalone


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run embedding component: load existing embeddings and compute missing ones per email."
    )
    parser.add_argument(
        "misp_path",
        nargs="?",
        help="Path to MISP JSON file. If omitted, only ensures output dir exists.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=f"Embeddings output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    args = parser.parse_args()

    out_dir = args.output_dir or str(DEFAULT_OUTPUT_DIR)
    if not args.misp_path:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        print(f"Output dir: {out_dir}")
        return 0

    subj_vecs, body_vecs, subj_dim, body_dim = run_standalone(args.misp_path, output_dir=out_dir)
    n = len(subj_vecs) if subj_vecs else len(body_vecs)
    print(f"Embeddings ready: {n} emails, subj_dim={subj_dim}, body_dim={body_dim}, output={out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
