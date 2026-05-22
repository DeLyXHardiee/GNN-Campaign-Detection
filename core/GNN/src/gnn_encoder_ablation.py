"""
Reversible GNN encoder ablation: reduced semantic email projection + relation edge dropout.

Controlled via pipeline_config ``gnn_encoder_ablation``; defaults preserve legacy behavior.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from torch import nn


def _edge_type_key(et: tuple[str, str, str]) -> str:
    return f"{et[0]},{et[1]},{et[2]}"


def parse_edge_type_key(key: str) -> tuple[str, str, str]:
    parts = [p.strip() for p in str(key).split(",")]
    if len(parts) != 3:
        raise ValueError(f"Invalid edge type key {key!r}; expected 'src,rel,dst'.")
    return parts[0], parts[1], parts[2]


@dataclass(frozen=True)
class GnnEncoderAblationConfig:
    """Training-time GNN / email-input ablation (off = legacy HeteroSAGE path)."""

    enabled: bool = False
    name: str = ""

    email_semantic_proj_dim: int = 32
    email_nonsemantic_proj_dim: int = 64
    email_semantic_feature_dropout: float = 0.30
    email_baked_semantic_dim: int = 128
    email_baked_nonsemantic_dim: int = 32

    gnn_edge_dropout: float = 0.20
    gnn_default_relation_dropout: float = 0.15
    gnn_relation_dropout_overrides: dict[str, float] = field(default_factory=dict)

    use_domain_relation: bool = True
    use_html_fp_relation: bool = True

    def to_log_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_gnn_encoder_ablation(raw: Any) -> GnnEncoderAblationConfig:
    if raw is None:
        return GnnEncoderAblationConfig()
    if not isinstance(raw, dict):
        raise TypeError("gnn_encoder_ablation must be an object or omitted.")
    overrides = raw.get("gnn_relation_dropout_overrides") or {}
    if not isinstance(overrides, dict):
        raise TypeError("gnn_relation_dropout_overrides must be an object.")
    return GnnEncoderAblationConfig(
        enabled=bool(raw.get("enabled", False)),
        name=str(raw.get("name") or ""),
        email_semantic_proj_dim=int(raw.get("email_semantic_proj_dim", 32)),
        email_nonsemantic_proj_dim=int(raw.get("email_nonsemantic_proj_dim", 64)),
        email_semantic_feature_dropout=float(raw.get("email_semantic_feature_dropout", 0.30)),
        email_baked_semantic_dim=int(raw.get("email_baked_semantic_dim", 128)),
        email_baked_nonsemantic_dim=int(raw.get("email_baked_nonsemantic_dim", 32)),
        gnn_edge_dropout=float(raw.get("gnn_edge_dropout", 0.20)),
        gnn_default_relation_dropout=float(raw.get("gnn_default_relation_dropout", 0.15)),
        gnn_relation_dropout_overrides={str(k): float(v) for k, v in overrides.items()},
        use_domain_relation=bool(raw.get("use_domain_relation", True)),
        use_html_fp_relation=bool(raw.get("use_html_fp_relation", True)),
    )


def relation_dropout_map(cfg: GnnEncoderAblationConfig) -> dict[tuple[str, str, str], float]:
    out: dict[tuple[str, str, str], float] = {}
    for k, p in (cfg.gnn_relation_dropout_overrides or {}).items():
        out[parse_edge_type_key(k)] = float(p)
    return out


class EmailGnnInputAdapter(nn.Module):
    """
    Re-project baked email.x [semantic | non-semantic] and apply dropout on semantic block only.
    """

    def __init__(
        self,
        *,
        semantic_in: int,
        semantic_out: int,
        other_in: int,
        other_out: int,
        semantic_dropout: float,
    ):
        super().__init__()
        if semantic_in <= 0 or other_in <= 0:
            raise ValueError("semantic_in and other_in must be positive.")
        self.semantic_in = int(semantic_in)
        self.other_in = int(other_in)
        self.semantic_out = int(semantic_out)
        self.other_out = int(other_out)
        self.sem_proj = (
            nn.Linear(self.semantic_in, self.semantic_out)
            if self.semantic_in != self.semantic_out
            else nn.Identity()
        )
        self.other_proj = (
            nn.Linear(self.other_in, self.other_out)
            if self.other_in != self.other_out
            else nn.Identity()
        )
        self.sem_drop = nn.Dropout(p=float(semantic_dropout))

    @property
    def out_dim(self) -> int:
        return self.semantic_out + self.other_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError("Expected 2D email features.")
        need = self.semantic_in + self.other_in
        if x.size(1) < need:
            raise ValueError(f"email.x dim {x.size(1)} < expected {need} for adapter split.")
        sem = x[:, : self.semantic_in]
        oth = x[:, self.semantic_in : self.semantic_in + self.other_in]
        sem = self.sem_proj(sem)
        if self.training:
            sem = self.sem_drop(sem)
        oth = self.other_proj(oth)
        return torch.cat([sem, oth], dim=1)


def apply_hetero_edge_dropout(
    edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
    *,
    default_p: float,
    relation_p: dict[tuple[str, str, str], float] | None = None,
    training: bool,
) -> dict[tuple[str, str, str], torch.Tensor]:
    """Randomly drop edges per relation; training only."""
    if not training or default_p <= 0:
        return edge_index_dict
    rel_p = relation_p or {}
    out: dict[tuple[str, str, str], torch.Tensor] = {}
    for et, ei in edge_index_dict.items():
        if ei is None or ei.numel() == 0:
            out[et] = ei
            continue
        p = float(rel_p.get(et, default_p))
        if p <= 0:
            out[et] = ei
            continue
        n = int(ei.size(1))
        keep = torch.rand(n, device=ei.device) > p
        if not bool(keep.any()):
            keep[0] = True
        out[et] = ei[:, keep]
    return out


def filter_edge_types_for_ablation(
    edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
    *,
    use_domain: bool,
    use_html_fp: bool,
) -> dict[tuple[str, str, str], torch.Tensor]:
    """Drop broad relations when ablation disables them (eval + train)."""
    out = dict(edge_index_dict)
    if not use_domain and ("email", "has_domain", "domain") in out:
        del out[("email", "has_domain", "domain")]
    if not use_html_fp and ("email", "has_html_structure_fingerprint", "html_structure_fingerprint") in out:
        del out[("email", "has_html_structure_fingerprint", "html_structure_fingerprint")]
    return out


def gnn_encoder_ablation_from_training_cfg(training_cfg: dict[str, Any]) -> GnnEncoderAblationConfig:
    return parse_gnn_encoder_ablation(training_cfg.get("gnn_encoder_ablation"))


def dump_gnn_encoder_ablation_summary(cfg: GnnEncoderAblationConfig) -> str:
    return json.dumps(cfg.to_log_dict(), indent=2, sort_keys=True)
