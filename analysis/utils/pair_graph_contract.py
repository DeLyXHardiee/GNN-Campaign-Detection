from __future__ import annotations

from typing import Any

import pandas as pd


GRAPH_KIND_ANCHOR = "anchor"
GRAPH_KIND_SEED = "seed"
GRAPH_KIND_CANDIDATE = "candidate"
GRAPH_KIND_SEED_CANDIDATE = "seed_candidate"
GRAPH_KIND_ALL: tuple[str, ...] = (
    GRAPH_KIND_ANCHOR,
    GRAPH_KIND_SEED,
    GRAPH_KIND_CANDIDATE,
    GRAPH_KIND_SEED_CANDIDATE,
)


REQUIRED_UNSCORED_COLUMNS: tuple[str, ...] = (
    "email_i",
    "email_j",
    "graph_kind",
    "graph_run_id",
    "from_seed",
    "from_semantic",
    "from_rare_artifact",
    "from_component",
    "from_2hop",
    "source_count",
)

REQUIRED_SCORED_COLUMNS: tuple[str, ...] = (
    "email_i",
    "email_j",
    "edge_weight",
    "score_mode",
)


def _pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def ensure_pairgraph_identity(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "email_i" not in out.columns or "email_j" not in out.columns:
        raise ValueError("PairGraph requires email_i and email_j")
    out["email_i"] = out["email_i"].astype(str)
    out["email_j"] = out["email_j"].astype(str)
    pairs = [_pair(a, b) for a, b in zip(out["email_i"], out["email_j"], strict=False)]
    out["email_i"] = [a for a, _ in pairs]
    out["email_j"] = [b for _, b in pairs]
    out = out[out["email_i"] != out["email_j"]].copy()
    out = out.drop_duplicates(subset=["email_i", "email_j"], keep="first").reset_index(drop=True)
    return out


def ensure_unscored_contract(df: pd.DataFrame) -> pd.DataFrame:
    out = ensure_pairgraph_identity(df)
    missing = [c for c in REQUIRED_UNSCORED_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required unscored PairGraph columns: {missing}")
    for c in ["from_seed", "from_semantic", "from_rare_artifact", "from_component", "from_2hop"]:
        out[c] = out[c].fillna(False).astype(bool)
    out["source_count"] = pd.to_numeric(out["source_count"], errors="coerce").fillna(0).astype(int)
    out["graph_kind"] = out["graph_kind"].astype(str)
    out["graph_run_id"] = out["graph_run_id"].astype(str)
    return out


def ensure_scored_contract(df: pd.DataFrame) -> pd.DataFrame:
    out = ensure_pairgraph_identity(df)
    missing = [c for c in REQUIRED_SCORED_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required scored PairGraph columns: {missing}")
    out["edge_weight"] = pd.to_numeric(out["edge_weight"], errors="coerce")
    out = out[out["edge_weight"].notna()].copy()
    out["score_mode"] = out["score_mode"].astype(str)
    return out


def pairgraph_meta(graph_kind: str, graph_run_id: str) -> dict[str, Any]:
    return {"graph_kind": str(graph_kind), "graph_run_id": str(graph_run_id)}


def validate_graph_kind(graph_kind: str) -> str:
    k = str(graph_kind).strip().lower()
    if k not in GRAPH_KIND_ALL:
        raise ValueError(f"Unsupported graph_kind {graph_kind!r}; expected one of {list(GRAPH_KIND_ALL)}")
    return k


def validate_score_mode_target_compatibility(*, score_mode: str, graph_kind: str) -> None:
    mode = str(score_mode).strip().lower()
    kind = validate_graph_kind(graph_kind)
    if mode in {"seed_candidate_handcrafted_v1", "seed_candidate_pu_v1"} and kind != GRAPH_KIND_SEED_CANDIDATE:
        raise ValueError(
            f"score_mode {mode!r} supports only graph_kind={GRAPH_KIND_SEED_CANDIDATE!r}, got {kind!r}"
        )
