from __future__ import annotations

from typing import Any

import pandas as pd


GRAPH_KIND_ANCHOR = "anchor"
GRAPH_KIND_SEED = "seed"
GRAPH_KIND_CANDIDATE = "candidate"
GRAPH_KIND_SEED_CANDIDATE = "seed_candidate"
GRAPH_KIND_SEMANTIC_SHARD = "semantic_shard"
GRAPH_KIND_ALL: tuple[str, ...] = (
    GRAPH_KIND_ANCHOR,
    GRAPH_KIND_SEED,
    GRAPH_KIND_CANDIDATE,
    GRAPH_KIND_SEED_CANDIDATE,
    GRAPH_KIND_SEMANTIC_SHARD,
)

# Legacy unscored PairGraph CSV column (pre-migration). Read paths normalize to ``graph_id``.
LEGACY_GRAPH_ID_COLUMN = "graph_run_id"

REQUIRED_UNSCORED_COLUMNS: tuple[str, ...] = (
    "email_i",
    "email_j",
    "graph_kind",
    "graph_id",
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


def migrate_unscored_graph_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy with legacy ``graph_run_id`` normalized to ``graph_id``.

    If both columns exist, ``graph_id`` is kept and ``graph_run_id`` is dropped.
    """
    out = df.copy()
    has_new = "graph_id" in out.columns
    has_old = LEGACY_GRAPH_ID_COLUMN in out.columns
    if has_new and has_old:
        return out.drop(columns=[LEGACY_GRAPH_ID_COLUMN])
    if has_old and not has_new:
        return out.rename(columns={LEGACY_GRAPH_ID_COLUMN: "graph_id"})
    return out


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
    out = migrate_unscored_graph_id_column(out)
    missing = [c for c in REQUIRED_UNSCORED_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required unscored PairGraph columns: {missing}")
    for c in ["from_seed", "from_semantic", "from_rare_artifact", "from_component", "from_2hop"]:
        out[c] = out[c].fillna(False).astype(bool)
    out["source_count"] = pd.to_numeric(out["source_count"], errors="coerce").fillna(0).astype(int)
    out["graph_kind"] = out["graph_kind"].astype(str)
    out["graph_id"] = out["graph_id"].astype(str)
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


def pairgraph_meta(graph_kind: str, graph_id: str) -> dict[str, Any]:
    return {"graph_kind": str(graph_kind), "graph_id": str(graph_id)}


def validate_graph_kind(graph_kind: str) -> str:
    k = str(graph_kind).strip().lower()
    if k not in GRAPH_KIND_ALL:
        raise ValueError(f"Unsupported graph_kind {graph_kind!r}; expected one of {list(GRAPH_KIND_ALL)}")
    return k


def validate_score_mode_target_compatibility(*, score_mode: str, graph_kind: str) -> None:
    mode = str(score_mode).strip().lower()
    kind = validate_graph_kind(graph_kind)
    if mode in {
        "seed_candidate_handcrafted_v1",
        "seed_candidate_pu_v1",
        "seed_candidate_edge_gnn_v1",
    } and kind != GRAPH_KIND_SEED_CANDIDATE:
        raise ValueError(
            f"score_mode {mode!r} supports only graph_kind={GRAPH_KIND_SEED_CANDIDATE!r}, got {kind!r}"
        )
    if mode in {"semantic_shard_handcrafted_v1", "semantic_shard_affine_v1"} and kind != GRAPH_KIND_SEMANTIC_SHARD:
        raise ValueError(
            f"score_mode {mode!r} supports only graph_kind={GRAPH_KIND_SEMANTIC_SHARD!r}, got {kind!r}"
        )
