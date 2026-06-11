"""Homogeneous GNN over candidate-edge nodes (line graph), not hetero email GNN."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv

LOCAL_HEAD_EDGE_GNN_DEFAULT = "edge_gnn_default"
LOCAL_HEAD_EMAIL_PAIR_MLP_COMPATIBLE = "email_pair_mlp_compatible"

COMBINE_MODE_GRAPH_ONLY = "graph_only"
COMBINE_MODE_CONCAT_LOCAL_GRAPH = "concat_local_graph"
COMBINE_MODES = (COMBINE_MODE_GRAPH_ONLY, COMBINE_MODE_CONCAT_LOCAL_GRAPH)


def edge_gnn_config_from_training_cfg(training_cfg: dict[str, Any]) -> dict[str, Any]:
    block = dict(training_cfg.get("edge_gnn") or {})
    local_head = str(block.get("local_head", LOCAL_HEAD_EDGE_GNN_DEFAULT)).strip().lower()
    if local_head not in (LOCAL_HEAD_EDGE_GNN_DEFAULT, LOCAL_HEAD_EMAIL_PAIR_MLP_COMPATIBLE):
        raise ValueError(
            f"Unsupported edge_gnn.local_head={local_head!r}; "
            f"use {LOCAL_HEAD_EDGE_GNN_DEFAULT!r} or {LOCAL_HEAD_EMAIL_PAIR_MLP_COMPATIBLE!r}."
        )
    combine_mode = str(block.get("combine_mode", COMBINE_MODE_CONCAT_LOCAL_GRAPH)).strip().lower()
    if combine_mode not in COMBINE_MODES:
        raise ValueError(
            f"Unsupported edge_gnn.combine_mode={combine_mode!r}; "
            f"use {COMBINE_MODE_GRAPH_ONLY!r} or {COMBINE_MODE_CONCAT_LOCAL_GRAPH!r}."
        )
    hidden = int(
        block.get("hidden_dim", training_cfg.get("pair_scorer_hidden_dim", training_cfg.get("hidden", 128)))
    )
    dropout = float(block.get("dropout", training_cfg.get("pair_scorer_dropout", 0.2)))
    return {
        "hidden_dim": hidden,
        "num_gnn_layers": int(block.get("num_gnn_layers", training_cfg.get("layers", 2))),
        "dropout": dropout,
        "local_head": local_head,
        "combine_mode": combine_mode,
        "max_neighbors_per_endpoint": block.get(
            "max_neighbors_per_endpoint",
            64,
        ),
        "rank_column": str(block.get("rank_column", "semantic_cosine_max")),
        "train_batch_size": int(
            block.get("train_batch_size") or training_cfg.get("pair_batch_size", 4096)
        ),
    }


def build_edge_pair_gnn_model(
    in_dim: int,
    training_cfg: dict[str, Any],
    *,
    edge_cfg: dict[str, Any] | None = None,
) -> "EdgePairGnnModel":
    """Construct EdgePairGnnModel from merged training config (incl. ``edge_gnn`` block)."""
    cfg = edge_cfg if edge_cfg is not None else edge_gnn_config_from_training_cfg(training_cfg)
    local_head = str(cfg["local_head"])
    num_gnn_layers = int(cfg["num_gnn_layers"])
    combine_mode = str(cfg.get("combine_mode", COMBINE_MODE_CONCAT_LOCAL_GRAPH))

    if local_head == LOCAL_HEAD_EMAIL_PAIR_MLP_COMPATIBLE:
        if num_gnn_layers == 0:
            from .pair_scorer import build_email_pair_mlp_scorer

            mlp_training_cfg = {
                **training_cfg,
                "pair_scorer_use_embedding_features": False,
                "pair_scorer_use_explicit_features": True,
                "pair_scorer_hidden_dim": int(cfg["hidden_dim"]),
                "pair_scorer_dropout": float(cfg["dropout"]),
            }
            pair_scorer = build_email_pair_mlp_scorer(
                embed_dim=1,
                pair_feat_dim=int(in_dim),
                training_cfg=mlp_training_cfg,
            )
            return EdgePairGnnModel(
                in_dim,
                num_gnn_layers=0,
                local_head=local_head,
                pair_scorer=pair_scorer,
            )
        return EdgePairGnnModel(
            in_dim,
            hidden_dim=int(cfg["hidden_dim"]),
            num_gnn_layers=num_gnn_layers,
            dropout=float(cfg["dropout"]),
            local_head=local_head,
            combine_mode=combine_mode,
        )

    return EdgePairGnnModel(
        in_dim,
        hidden_dim=int(cfg["hidden_dim"]),
        num_gnn_layers=num_gnn_layers,
        dropout=float(cfg["dropout"]),
        local_head=local_head,
        combine_mode=combine_mode,
    )


class EdgePairGnnModel(nn.Module):
    """
    Candidate-edge node model: pair features -> optional GraphSAGE -> scalar logit.

    ``num_gnn_layers=0`` skips message passing. With ``local_head=email_pair_mlp_compatible``
    and ``num_gnn_layers=0``, reuses ``EmailPairMLPScorer`` (same as ``_14``).

    With ``local_head=email_pair_mlp_compatible`` and ``num_gnn_layers>=1``:

    ``h_local = Linear→ReLU→Dropout`` then GraphSAGE on ``h_local``, then
    ``concat(h_local, h_graph)`` (default) or graph-only logits.
    """

    def __init__(
        self,
        in_dim: int,
        *,
        hidden_dim: int = 128,
        gnn_hidden_dim: int | None = None,
        num_gnn_layers: int = 2,
        dropout: float = 0.2,
        local_head: str = LOCAL_HEAD_EDGE_GNN_DEFAULT,
        combine_mode: str = COMBINE_MODE_CONCAT_LOCAL_GRAPH,
        pair_scorer: nn.Module | None = None,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.local_head = str(local_head).strip().lower()
        self.num_gnn_layers = max(0, int(num_gnn_layers))
        self.dropout_p = float(dropout)
        self.combine_mode = str(combine_mode).strip().lower()
        self.pair_scorer = pair_scorer

        if self.pair_scorer is not None:
            if self.num_gnn_layers != 0:
                raise ValueError("pair_scorer is only valid with num_gnn_layers=0")
            if self.local_head != LOCAL_HEAD_EMAIL_PAIR_MLP_COMPATIBLE:
                raise ValueError("pair_scorer requires local_head=email_pair_mlp_compatible")
            self.hidden_dim = int(getattr(pair_scorer, "_in_dim", in_dim))
            self.local_encoder = None
            self.convs = nn.ModuleList()
            self.output_mlp = None
            return

        self.hidden_dim = int(gnn_hidden_dim if gnn_hidden_dim is not None else hidden_dim)

        if self.local_head == LOCAL_HEAD_EMAIL_PAIR_MLP_COMPATIBLE:
            self.local_encoder = nn.Sequential(
                nn.Linear(self.in_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
            )
            self.input_mlp = None
        else:
            self.local_encoder = None
            self.input_mlp = nn.Sequential(
                nn.Linear(self.in_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
            )

        self.convs = nn.ModuleList()
        for _ in range(self.num_gnn_layers):
            self.convs.append(SAGEConv((self.hidden_dim, self.hidden_dim), self.hidden_dim))

        if self.num_gnn_layers == 0:
            out_in = self.hidden_dim
        elif (
            self.local_head == LOCAL_HEAD_EMAIL_PAIR_MLP_COMPATIBLE
            and self.combine_mode == COMBINE_MODE_CONCAT_LOCAL_GRAPH
        ):
            out_in = self.hidden_dim * 2
        else:
            out_in = self.hidden_dim
        self.output_mlp = nn.Linear(out_in, 1)

    @property
    def uses_mlp_compatible_local_graph(self) -> bool:
        return (
            self.local_head == LOCAL_HEAD_EMAIL_PAIR_MLP_COMPATIBLE
            and self.num_gnn_layers > 0
            and self.local_encoder is not None
        )

    def _local_hidden(self, x: torch.Tensor) -> torch.Tensor:
        if self.local_encoder is not None:
            return self.local_encoder(x)
        if self.input_mlp is not None:
            return self.input_mlp(x)
        raise RuntimeError("model has no local encoder")

    def _message_pass(self, h_local: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = h_local
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if i < len(self.convs) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout_p, training=self.training)
        return h

    def forward_hidden(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return ``(logits, h_graph)``; ``h_graph`` is None when no message passing."""
        if self.pair_scorer is not None:
            b = int(x.size(0))
            z_dummy = torch.zeros((b, 1), device=x.device, dtype=x.dtype)
            logits = self.pair_scorer(z_dummy, z_dummy, x)
            return logits, None

        h_local = self._local_hidden(x)
        if self.num_gnn_layers == 0:
            return self.output_mlp(h_local).squeeze(-1), None

        h_graph = self._message_pass(h_local, edge_index)
        if (
            self.local_head == LOCAL_HEAD_EMAIL_PAIR_MLP_COMPATIBLE
            and self.combine_mode == COMBINE_MODE_CONCAT_LOCAL_GRAPH
        ):
            h_combined = torch.cat([h_local, h_graph], dim=-1)
            return self.output_mlp(h_combined).squeeze(-1), h_graph
        return self.output_mlp(h_graph).squeeze(-1), h_graph

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.size(1) != self.in_dim:
            raise ValueError(f"Expected x shape (N, {self.in_dim}), got {tuple(x.shape)}")
        logits, _ = self.forward_hidden(x, edge_index)
        return logits

    @torch.no_grad()
    def graph_representation_stats(
        self, x: torch.Tensor, edge_index: torch.Tensor, mask: torch.Tensor | None = None
    ) -> dict[str, float]:
        """L2 norm mean/std of local vs graph hidden states (diagnostic)."""
        if not self.uses_mlp_compatible_local_graph:
            return {}
        was_training = self.training
        self.eval()
        h_local = self._local_hidden(x)
        h_graph = self._message_pass(h_local, edge_index)
        if mask is not None and mask.any():
            h_local = h_local[mask]
            h_graph = h_graph[mask]
        local_norm = h_local.norm(dim=-1)
        graph_norm = h_graph.norm(dim=-1)
        out = {
            "h_local_norm_mean": float(local_norm.mean().item()),
            "h_local_norm_std": float(local_norm.std(unbiased=False).item()) if local_norm.numel() > 1 else 0.0,
            "h_graph_norm_mean": float(graph_norm.mean().item()),
            "h_graph_norm_std": float(graph_norm.std(unbiased=False).item()) if graph_norm.numel() > 1 else 0.0,
            "graph_delta_norm_mean": float((h_graph - h_local).norm(dim=-1).mean().item()),
        }
        if was_training:
            self.train()
        return out
