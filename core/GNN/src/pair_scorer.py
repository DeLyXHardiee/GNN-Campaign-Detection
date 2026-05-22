"""
Substep 4: dedicated email-email pair scorer (pair-supervision path only).

Builds representation from z_i, z_j, |z_i - z_j|, z_i * z_j, optional explicit pair metadata.
Outputs one scalar logit per row. Not used for graph-native link prediction.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class EmailPairMLPScorer(nn.Module):
    """
    MLP on [z_i, z_j, |z_i - z_j|, z_i ⊙ z_j, pair_feats?] -> (B,) logits.

    ``pair_feats`` omitted when ``use_explicit_pair_features=False`` (in_dim = 4 * embed_dim).
    """

    def __init__(
        self,
        embed_dim: int,
        *,
        pair_feat_dim: int = 0,
        mlp_hidden_dim: int = 256,
        dropout: float = 0.2,
        use_explicit_pair_features: bool = True,
        use_embedding_features: bool = True,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.pair_feat_dim = int(pair_feat_dim) if use_explicit_pair_features else 0
        self.use_explicit_pair_features = bool(use_explicit_pair_features)
        self.use_embedding_features = bool(use_embedding_features)
        if not self.use_embedding_features and not self.use_explicit_pair_features:
            raise ValueError(
                "EmailPairMLPScorer requires at least one of use_embedding_features or "
                "use_explicit_pair_features."
            )
        if self.use_embedding_features:
            in_dim = 4 * self.embed_dim + (
                self.pair_feat_dim if use_explicit_pair_features else 0
            )
        else:
            in_dim = self.pair_feat_dim
        self._in_dim = in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, int(mlp_hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(mlp_hidden_dim), 1),
        )

    @property
    def input_feature_dim(self) -> int:
        return self._in_dim

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor, pair_feats: torch.Tensor | None) -> torch.Tensor:
        if not self.use_embedding_features:
            if pair_feats is None:
                raise ValueError("pair_feats required when use_embedding_features=False")
            b = int(pair_feats.size(0))
            if pair_feats.dim() != 2 or pair_feats.size(1) != self.pair_feat_dim:
                raise ValueError(
                    f"pair_feats must be (B, {self.pair_feat_dim}), got {pair_feats.shape}"
                )
            return self.net(pair_feats).squeeze(-1)

        if z_i.shape != z_j.shape:
            raise ValueError(f"z_i/z_j shape mismatch: {z_i.shape} vs {z_j.shape}")
        b, d = z_i.shape[0], z_i.shape[1]
        if d != self.embed_dim:
            raise ValueError(f"embed_dim mismatch: model {d} vs scorer {self.embed_dim}")
        parts = [z_i, z_j, (z_i - z_j).abs(), z_i * z_j]
        if self.use_explicit_pair_features:
            if pair_feats is None:
                raise ValueError("pair_feats required when use_explicit_pair_features=True")
            if pair_feats.dim() != 2 or pair_feats.size(0) != b:
                raise ValueError(f"pair_feats must be (B, F) with B={b}, got {None if pair_feats is None else pair_feats.shape}")
            if pair_feats.size(1) != self.pair_feat_dim:
                raise ValueError(f"pair_feats dim {pair_feats.size(1)} != expected {self.pair_feat_dim}")
            parts.append(pair_feats)
        elif pair_feats is not None:
            raise ValueError("pair_feats passed but use_explicit_pair_features=False")
        x = torch.cat(parts, dim=-1)
        return self.net(x).squeeze(-1)


def build_email_pair_mlp_scorer(
    embed_dim: int,
    pair_feat_dim: int,
    training_cfg: dict[str, Any],
) -> EmailPairMLPScorer:
    """
    Factory from merged training / pair_training config.

    ``pair_scorer_type``: ``email_pair_mlp`` (default) or ``mlp_concat`` (alias).
    """
    scorer_type = str(training_cfg.get("pair_scorer_type", "email_pair_mlp")).lower().strip()
    if scorer_type not in ("email_pair_mlp", "mlp_concat"):
        raise ValueError(
            f"Unsupported pair_scorer_type={scorer_type!r}; use 'email_pair_mlp' or 'mlp_concat'."
        )
    use_explicit = bool(training_cfg.get("pair_scorer_use_explicit_features", True))
    use_embedding = bool(training_cfg.get("pair_scorer_use_embedding_features", True))
    hidden = int(training_cfg.get("pair_scorer_hidden_dim", 256))
    drop = float(training_cfg.get("pair_scorer_dropout", 0.2))
    pf = int(pair_feat_dim) if use_explicit else 0
    if use_explicit and pair_feat_dim < 0:
        raise ValueError("pair_feat_dim must be non-negative.")
    if not use_embedding and not use_explicit:
        raise ValueError(
            "pair_scorer_use_embedding_features=False requires pair_scorer_use_explicit_features=True."
        )
    return EmailPairMLPScorer(
        embed_dim=int(embed_dim),
        pair_feat_dim=pf,
        mlp_hidden_dim=hidden,
        dropout=drop,
        use_explicit_pair_features=use_explicit,
        use_embedding_features=use_embedding,
    )


def count_scorer_parameters(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


# Backward-compatible name (same class).
PairScorer = EmailPairMLPScorer
