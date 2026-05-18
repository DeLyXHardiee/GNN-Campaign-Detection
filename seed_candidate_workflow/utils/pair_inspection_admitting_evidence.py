"""
Load per-family candidate source rows and format admitting evidence for pair inspection HTML/CSV.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Normalized record keys written to admitting_evidence_json / HTML.
_ADMIT_KEYS = (
    "source_family",
    "artifact_type",
    "artifact_value",
    "path_type",
    "rarity_score",
    "artifact_degree",
    "cosine",
    "reason_code",
    "rule_id",
    "seed_generator",
    "evidence_type",
    "support_type",
    "proposal_type",
    "rank_i_to_j",
    "rank_j_to_i",
    "mutual_topk",
    "n_shared_core_channels",
    "extra",
)


def pair_key(a: str, b: str) -> tuple[str, str]:
    a, b = str(a).strip(), str(b).strip()
    return (a, b) if a <= b else (b, a)


def resolve_candidate_generation_dir(
    *,
    pair_training_csv: Path,
    explicit: Path | None = None,
) -> Path | None:
    if explicit is not None:
        p = explicit.expanduser().resolve()
        return p if p.is_dir() else None
    pair_training_csv = pair_training_csv.expanduser().resolve()
    if not pair_training_csv.is_file():
        return None
    graph_id = pair_training_csv.parent.name
    bundle_root = pair_training_csv.parent.parent.parent
    cand_parent = bundle_root / "candidate" / graph_id
    if not cand_parent.is_dir():
        return None
    gens = sorted(
        [d for d in cand_parent.iterdir() if d.is_dir() and d.name.startswith("candidate_generation")],
        key=lambda d: d.stat().st_mtime,
    )
    return gens[-1] if gens else None


def resolve_seed_generation_dir(*, pair_training_csv: Path) -> Path | None:
    pair_training_csv = pair_training_csv.expanduser().resolve()
    graph_id = pair_training_csv.parent.name
    bundle_root = pair_training_csv.parent.parent.parent
    seed_parent = bundle_root / "seed" / graph_id
    if not seed_parent.is_dir():
        return None
    gens = sorted(
        [d for d in seed_parent.iterdir() if d.is_dir() and d.name.startswith("seed_generation")],
        key=lambda d: d.stat().st_mtime,
    )
    return gens[-1] if gens else None


def _norm_record(**kwargs: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in _ADMIT_KEYS:
        v = kwargs.get(k)
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            continue
        if isinstance(v, str) and not v.strip():
            continue
        out[k] = v
    extra = kwargs.get("extra")
    if isinstance(extra, dict) and extra:
        out["extra"] = extra
    return out


def _append_index(index: dict[tuple[str, str], list[dict[str, Any]]], pk: tuple[str, str], rec: dict[str, Any]) -> None:
    if not rec:
        return
    index.setdefault(pk, []).append(rec)


def _load_csv_rows(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except (OSError, pd.errors.ParserError, ValueError):
        return pd.DataFrame()


def _ingest_2hop(df: pd.DataFrame, index: dict[tuple[str, str], list[dict[str, Any]]]) -> int:
    if df.empty or "email_i" not in df.columns:
        return 0
    n = 0
    for _, r in df.iterrows():
        pk = pair_key(r["email_i"], r["email_j"])
        _append_index(
            index,
            pk,
            _norm_record(
                source_family="2hop",
                artifact_type=str(r.get("intermediary_artifact_type") or ""),
                artifact_value=str(r.get("intermediary_artifact_value") or ""),
                path_type=str(r.get("path_type") or ""),
                rarity_score=pd.to_numeric(r.get("rarity_score"), errors="coerce"),
                artifact_degree=pd.to_numeric(r.get("intermediary_degree"), errors="coerce"),
                reason_code=str(r.get("reason_code") or ""),
                extra={
                    "seed_adjacent_flag": bool(r.get("seed_adjacent_flag", False)),
                    "both_in_seed_components": bool(r.get("both_in_seed_components", False)),
                },
            ),
        )
        n += 1
    return n


def _ingest_rare_like(df: pd.DataFrame, index: dict[tuple[str, str], list[dict[str, Any]]], family: str) -> int:
    if df.empty or "email_i" not in df.columns:
        return 0
    n = 0
    for _, r in df.iterrows():
        pk = pair_key(r["email_i"], r["email_j"])
        _append_index(
            index,
            pk,
            _norm_record(
                source_family=family,
                artifact_type=str(r.get("artifact_type") or ""),
                artifact_value=str(r.get("artifact_value") or ""),
                rarity_score=pd.to_numeric(r.get("rarity_score"), errors="coerce"),
                artifact_degree=pd.to_numeric(r.get("artifact_df"), errors="coerce"),
            ),
        )
        n += 1
    return n


def _ingest_semantic(df: pd.DataFrame, index: dict[tuple[str, str], list[dict[str, Any]]], family: str) -> int:
    if df.empty or "email_i" not in df.columns:
        return 0
    n = 0
    for _, r in df.iterrows():
        pk = pair_key(r["email_i"], r["email_j"])
        _append_index(
            index,
            pk,
            _norm_record(
                source_family=family,
                cosine=pd.to_numeric(r.get("cosine"), errors="coerce"),
                rank_i_to_j=pd.to_numeric(r.get("rank_i_to_j"), errors="coerce"),
                rank_j_to_i=pd.to_numeric(r.get("rank_j_to_i"), errors="coerce"),
                mutual_topk=bool(r.get("mutual_topk", False)) if "mutual_topk" in r.index else None,
                n_shared_core_channels=pd.to_numeric(r.get("n_shared_core_channels"), errors="coerce"),
                extra={
                    "has_shared_sender": bool(r.get("has_shared_sender", False)),
                    "has_shared_stem": bool(r.get("has_shared_stem", False)),
                    "semantic_min_cos": r.get("semantic_min_cos"),
                    "semantic_max_cos_exclusive": r.get("semantic_max_cos_exclusive"),
                },
            ),
        )
        n += 1
    return n


def _ingest_component(df: pd.DataFrame, index: dict[tuple[str, str], list[dict[str, Any]]]) -> int:
    if df.empty or "email_i" not in df.columns:
        return 0
    n = 0
    for _, r in df.iterrows():
        pk = pair_key(r["email_i"], r["email_j"])
        art_col = str(r.get("artifact_col") or "").strip()
        art_type = art_col.replace("_set", "") if art_col else ""
        _append_index(
            index,
            pk,
            _norm_record(
                source_family="component",
                support_type=str(r.get("support_type") or ""),
                cosine=pd.to_numeric(r.get("cosine"), errors="coerce"),
                artifact_type=art_type,
                artifact_value=str(r.get("artifact_value") or ""),
                proposal_type=str(r.get("proposal_type") or ""),
                reason_code=str(r.get("reason_code") or ""),
                extra={
                    "component_a": r.get("component_a"),
                    "component_b": r.get("component_b"),
                },
            ),
        )
        n += 1
    return n


def _ingest_seed(df: pd.DataFrame, index: dict[tuple[str, str], list[dict[str, Any]]]) -> int:
    if df.empty or "email_i" not in df.columns:
        return 0
    n = 0
    for _, r in df.iterrows():
        pk = pair_key(r["email_i"], r["email_j"])
        _append_index(
            index,
            pk,
            _norm_record(
                source_family="seed",
                seed_generator=str(r.get("seed_generator") or ""),
                evidence_type=str(r.get("evidence_type") or ""),
                rule_id=str(r.get("rule_id") or ""),
                artifact_value=str(r.get("evidence_value") or "")[:500],
                rarity_score=pd.to_numeric(r.get("evidence_rarity"), errors="coerce"),
            ),
        )
        n += 1
    return n


def load_admitting_evidence_index(
    *,
    candidate_generation_dir: Path | None,
    seed_generation_dir: Path | None,
) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], dict[str, Any]]:
    """Pair key -> sorted list of normalized admitting-evidence records."""
    index: dict[tuple[str, str], list[dict[str, Any]]] = {}
    meta: dict[str, Any] = {"status": "skipped", "source_files": {}}
    if candidate_generation_dir is None or not candidate_generation_dir.is_dir():
        meta["reason"] = "candidate_generation_dir_not_found"
        return index, meta

    cand_dir = candidate_generation_dir.resolve()
    loaders: list[tuple[str, Path, Any]] = [
        ("candidates_2hop.csv", cand_dir / "candidates_2hop.csv", lambda df: _ingest_2hop(df, index)),
        ("candidates_rare_artifact.csv", cand_dir / "candidates_rare_artifact.csv", lambda df: _ingest_rare_like(df, index, "rare_artifact")),
        (
            "candidates_shared_stem_highconf.csv",
            cand_dir / "candidates_shared_stem_highconf.csv",
            lambda df: _ingest_rare_like(df, index, "shared_stem_highconf"),
        ),
        ("candidates_semantic.csv", cand_dir / "candidates_semantic.csv", lambda df: _ingest_semantic(df, index, "semantic")),
        (
            "candidates_semantic_mid_sender_support.csv",
            cand_dir / "candidates_semantic_mid_sender_support.csv",
            lambda df: _ingest_semantic(df, index, "semantic_mid_sender"),
        ),
        ("candidates_mid_sender.csv", cand_dir / "candidates_mid_sender.csv", lambda df: _ingest_semantic(df, index, "semantic_mid_sender")),
        (
            "candidates_semantic_mid_core_support.csv",
            cand_dir / "candidates_semantic_mid_core_support.csv",
            lambda df: _ingest_semantic(df, index, "semantic_mid_core"),
        ),
        ("candidates_mid_core.csv", cand_dir / "candidates_mid_core.csv", lambda df: _ingest_semantic(df, index, "semantic_mid_core")),
        (
            "candidates_semantic_mid_stem_support.csv",
            cand_dir / "candidates_semantic_mid_stem_support.csv",
            lambda df: _ingest_semantic(df, index, "semantic_mid_stem"),
        ),
        ("candidates_mid_stem.csv", cand_dir / "candidates_mid_stem.csv", lambda df: _ingest_semantic(df, index, "semantic_mid_stem")),
        (
            "candidates_component_expanded.csv",
            cand_dir / "candidates_component_expanded.csv",
            _ingest_component,
        ),
    ]
    total_rows = 0
    for label, path, fn in loaders:
        df = _load_csv_rows(path)
        if df.empty:
            continue
        n = int(fn(df))
        total_rows += n
        meta["source_files"][label] = {"path": str(path), "n_rows_ingested": n}

    if seed_generation_dir is not None:
        seed_csv = seed_generation_dir.resolve() / "seed_edges_all.csv"
        df_seed = _load_csv_rows(seed_csv)
        n_seed = _ingest_seed(df_seed, index)
        total_rows += n_seed
        meta["source_files"]["seed_edges_all.csv"] = {
            "path": str(seed_csv),
            "n_rows_ingested": n_seed,
        }

    for pk, recs in index.items():
        recs.sort(key=_evidence_sort_key, reverse=True)

    meta.update(
        {
            "status": "ok",
            "candidate_generation_dir": str(cand_dir),
            "seed_generation_dir": str(seed_generation_dir.resolve()) if seed_generation_dir else None,
            "n_pairs_with_evidence": int(len(index)),
            "n_source_rows_ingested": int(total_rows),
        }
    )
    return index, meta


def _evidence_sort_key(rec: dict[str, Any]) -> tuple[float, float, str]:
    rarity = pd.to_numeric(rec.get("rarity_score"), errors="coerce")
    cos = pd.to_numeric(rec.get("cosine"), errors="coerce")
    r = float(rarity) if pd.notna(rarity) else -1.0
    c = float(cos) if pd.notna(cos) else -1.0
    return (r, c, str(rec.get("source_family") or ""))


def _truncate_val(val: str, max_len: int = 120) -> str:
    s = str(val or "").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def format_admitting_line(rec: dict[str, Any]) -> str:
    fam = str(rec.get("source_family") or "?")
    if fam == "2hop":
        at = str(rec.get("artifact_type") or "artifact")
        av = _truncate_val(str(rec.get("artifact_value") or ""))
        path = str(rec.get("path_type") or "")
        parts = [f"2hop via {at} = {av}"]
        if path:
            parts.append(f"path={path}")
        deg = rec.get("artifact_degree")
        if deg is not None and pd.notna(deg):
            parts.append(f"degree={int(deg)}")
        rs = rec.get("rarity_score")
        if rs is not None and pd.notna(rs):
            parts.append(f"rarity={float(rs):.3f}")
        rc = str(rec.get("reason_code") or "")
        if rc:
            parts.append(f"reason={rc}")
        return " · ".join(parts)
    if fam in {"rare_artifact", "shared_stem_highconf"}:
        at = str(rec.get("artifact_type") or fam)
        av = _truncate_val(str(rec.get("artifact_value") or ""))
        rs = rec.get("rarity_score")
        tail = f" (rarity={float(rs):.3f})" if rs is not None and pd.notna(rs) else ""
        df = rec.get("artifact_degree")
        if df is not None and pd.notna(df):
            tail += f" df={int(df)}"
        return f"{fam} via {at} = {av}{tail}"
    if fam.startswith("semantic"):
        cos = rec.get("cosine")
        cos_s = f"{float(cos):.4f}" if cos is not None and pd.notna(cos) else "?"
        mutual = rec.get("mutual_topk")
        rule = "mutual top-k reciprocal" if fam == "semantic" and mutual else fam.replace("_", " ")
        ri = rec.get("rank_i_to_j")
        rj = rec.get("rank_j_to_i")
        ranks = ""
        if ri is not None and pd.notna(ri) and rj is not None and pd.notna(rj):
            ranks = f" ranks {int(ri)}/{int(rj)}"
        ncore = rec.get("n_shared_core_channels")
        core_s = f" · {int(ncore)} shared core channels" if ncore is not None and pd.notna(ncore) else ""
        return f"{rule}: cosine={cos_s}{ranks}{core_s}"
    if fam == "component":
        st = str(rec.get("support_type") or "expansion")
        cos = rec.get("cosine")
        cos_s = f" cos={float(cos):.4f}" if cos is not None and pd.notna(cos) else ""
        av = _truncate_val(str(rec.get("artifact_value") or ""))
        at = str(rec.get("artifact_type") or "")
        art = f" · {at}={av}" if av and at else (f" · {av}" if av else "")
        pt = str(rec.get("proposal_type") or "")
        prop = f" · {pt}" if pt else ""
        return f"component {st}{cos_s}{art}{prop}"
    if fam == "seed":
        gen = str(rec.get("seed_generator") or "seed")
        rule = str(rec.get("rule_id") or rec.get("evidence_type") or "")
        ev = _truncate_val(str(rec.get("artifact_value") or ""), 80)
        bits = [f"seed ({gen})"]
        if rule:
            bits.append(rule)
        if ev:
            bits.append(ev)
        return ": ".join(bits)
    return f"{fam}: {json.dumps(rec, default=str)[:200]}"


def direct_shared_lines_from_row(row: pd.Series) -> list[str]:
    """Explicit direct overlaps from inspection row / anchor node join."""
    lines: list[str] = []
    channel_specs = (
        ("sender", "shared_sender_values"),
        ("sender_domain", "shared_sender_domain_values"),
        ("url", "shared_url_values"),
        ("stem", "shared_stem_values"),
        ("attachment", "shared_attachment_values"),
        ("domain", "shared_domain_values"),
        ("html_fp", "shared_html_fp_values"),
        ("received_host", "shared_received_host_values"),
    )
    for label, val_col in channel_specs:
        if bool(row.get(f"has_shared_{label}", False)):
            vals = str(row.get(val_col) or "").strip()
            if vals:
                lines.append(f"direct shared {label}: {_truncate_val(vals, 200)}")
            else:
                lines.append(f"direct shared {label}")
    return lines


def compute_inspection_warnings(row: pd.Series) -> list[str]:
    warnings: list[str] = []
    prov_parts = [
        bool(row.get("from_seed")),
        bool(row.get("from_semantic")),
        bool(row.get("from_rare_artifact")),
        bool(row.get("from_component")),
        bool(row.get("from_2hop")),
    ]
    n_prov = sum(prov_parts)
    sc = pd.to_numeric(row.get("source_count"), errors="coerce")
    if pd.notna(sc) and int(sc) == 1:
        warnings.append("source_count==1")
    if n_prov == 1 and bool(row.get("from_2hop")):
        warnings.append("2hop only")
    sem = pd.to_numeric(row.get("semantic_cosine_for_display"), errors="coerce")
    if pd.isna(sem):
        sem = pd.to_numeric(row.get("semantic_cosine_max"), errors="coerce")
    if pd.notna(sem) and float(sem) < 0.85:
        warnings.append("semantic<0.85")
    if bool(row.get("from_2hop")) and not bool(row.get("has_shared_sender")):
        warnings.append("2hop & not shared_sender")
    if bool(row.get("has_shared_domain")) and not any(
        bool(row.get(f"has_shared_{c}")) for c in ("sender", "url", "stem", "attachment")
    ):
        warnings.append("shared domain only")
    stem_vals = str(row.get("shared_stem_values") or "").strip()
    if stem_vals in {"/", "|/|", "/"} or stem_vals.split("|") == ["/"]:
        warnings.append("trivial stem '/'")
    return warnings


def enrich_inspection_with_admitting_evidence(
    df: pd.DataFrame,
    *,
    evidence_index: dict[tuple[str, str], list[dict[str, Any]]],
    max_lines_per_pair: int = 12,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()
    admitting_json: list[str] = []
    admitting_lines: list[str] = []
    direct_lines_col: list[str] = []
    combined_brief: list[str] = []
    warnings_col: list[str] = []

    for _, r in out.iterrows():
        pk = pair_key(r["email_i"], r["email_j"])
        recs = list(evidence_index.get(pk) or [])
        if bool(r.get("from_2hop")) and not any(str(x.get("source_family")) == "2hop" for x in recs):
            recs.insert(
                0,
                _norm_record(
                    source_family="2hop",
                    artifact_type="(missing from candidates_2hop.csv)",
                    artifact_value="re-run candidate generation or check graph bundle path",
                    extra={"synthetic_warning": True},
                ),
            )
        lines = [format_admitting_line(x) for x in recs[: max(1, max_lines_per_pair)]]
        direct = direct_shared_lines_from_row(r)
        warns = compute_inspection_warnings(r)

        admitting_json.append(json.dumps(recs[:max_lines_per_pair], ensure_ascii=False, default=str))
        admitting_lines.append("\n".join(lines))
        direct_lines_col.append("\n".join(direct))
        warnings_col.append("|".join(warns))

        brief_parts: list[str] = []
        if direct:
            brief_parts.extend(direct[:4])
        if lines:
            brief_parts.extend(lines[:6])
        combined_brief.append(" || ".join(brief_parts) if brief_parts else "none")

    out["admitting_evidence_json"] = admitting_json
    out["admitting_evidence_lines"] = admitting_lines
    out["direct_shared_evidence_lines"] = direct_lines_col
    out["shared_evidence_brief"] = combined_brief
    out["inspection_warning_flags"] = warnings_col
    # Back-compat: richer brief for meta row + old column name
    out["shared_artifacts_brief"] = combined_brief
    out = _attach_twohop_channel_columns_after_enrich(out, evidence_index=evidence_index)
    return out


def _attach_twohop_channel_columns_after_enrich(
    df: pd.DataFrame,
    *,
    evidence_index: dict[tuple[str, str], list[dict[str, Any]]],
) -> pd.DataFrame:
    from seed_candidate_workflow.utils.pair_low_band_twohop_channel import attach_twohop_channel_columns

    out = attach_twohop_channel_columns(df, evidence_index=evidence_index)
    if "twohop_channel_badges" in out.columns and "inspection_warning_flags" in out.columns:
        merged_warn: list[str] = []
        for _, r in out.iterrows():
            parts = [str(r.get("inspection_warning_flags") or "").strip()]
            ch_badges = str(r.get("twohop_channel_badges") or "").strip()
            if ch_badges:
                parts.append(ch_badges)
            merged_warn.append("|".join(p for p in parts if p))
        out["inspection_warning_flags"] = merged_warn
    return out
