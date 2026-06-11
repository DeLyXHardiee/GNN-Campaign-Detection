from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv, to_hetero

class SAGEBackbone(nn.Module):
    def __init__(self, hidden=128, out=128, layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv((-1, -1), hidden))
        for _ in range(layers - 2):
            self.layers.append(SAGEConv((hidden, hidden), hidden))
        if layers > 1:
            self.layers.append(SAGEConv((hidden, hidden), out))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        h = x
        for i, conv in enumerate(self.layers):
            h = conv(h, edge_index)
            if i < len(self.layers) - 1:
                h = torch.relu(h)
                h = self.dropout(h)
        h = F.normalize(h, p=2, dim=-1)
        return h


class HeteroSAGE(nn.Module):
    def __init__(self, metadata, hidden=128, out=128, layers=2, dropout=0.1):
        super().__init__()
        self.gnn = to_hetero(SAGEBackbone(hidden, out, layers, dropout), metadata)

    def forward(self, x_dict, edge_index_dict):
        return self.gnn(x_dict, edge_index_dict)


class PairGnnEncoder(nn.Module):
    """
    HeteroSAGE with optional email-input adapter and training-time relation edge dropout.
    """

    def __init__(
        self,
        metadata,
        hidden: int = 128,
        out: int = 128,
        layers: int = 2,
        dropout: float = 0.1,
        *,
        ablation: Any | None = None,
    ):
        super().__init__()
        from .gnn_encoder_ablation import (
            EmailGnnInputAdapter,
            GnnEncoderAblationConfig,
            apply_hetero_edge_dropout,
            filter_edge_types_for_ablation,
            relation_dropout_map,
        )

        self.metadata = metadata
        self.ablation: GnnEncoderAblationConfig = ablation or GnnEncoderAblationConfig()
        self.sage = HeteroSAGE(metadata=metadata, hidden=hidden, out=out, layers=layers, dropout=dropout)
        self.email_adapter: EmailGnnInputAdapter | None = None
        if self.ablation.enabled:
            self.email_adapter = EmailGnnInputAdapter(
                semantic_in=self.ablation.email_baked_semantic_dim,
                semantic_out=self.ablation.email_semantic_proj_dim,
                other_in=self.ablation.email_baked_nonsemantic_dim,
                other_out=self.ablation.email_nonsemantic_proj_dim,
                semantic_dropout=self.ablation.email_semantic_feature_dropout,
            )
        self._apply_hetero_edge_dropout = apply_hetero_edge_dropout
        self._filter_edges = filter_edge_types_for_ablation
        self._relation_dropout_map = relation_dropout_map

    def forward(self, x_dict, edge_index_dict):
        x_work = dict(x_dict)
        if self.email_adapter is not None and "email" in x_work:
            x_work["email"] = self.email_adapter(x_work["email"])
        e_work = edge_index_dict
        if self.ablation.enabled:
            e_work = self._filter_edges(
                e_work,
                use_domain=self.ablation.use_domain_relation,
                use_html_fp=self.ablation.use_html_fp_relation,
            )
            rel_p = self._relation_dropout_map(self.ablation)
            default_p = (
                float(self.ablation.gnn_edge_dropout)
                if self.ablation.gnn_edge_dropout > 0
                else float(self.ablation.gnn_default_relation_dropout)
            )
            e_work = self._apply_hetero_edge_dropout(
                e_work,
                default_p=default_p,
                relation_p=rel_p,
                training=self.training,
            )
        return self.sage(x_work, e_work)


def build_pair_gnn_encoder(
    metadata,
    training_cfg: dict,
    *,
    device: torch.device | None = None,
    state_dict: dict | None = None,
) -> PairGnnEncoder | HeteroSAGE:
    """Build GNN encoder; uses PairGnnEncoder when gnn_encoder_ablation.enabled."""
    from .gnn_encoder_ablation import gnn_encoder_ablation_from_training_cfg

    hidden = int(training_cfg.get("hidden", 128))
    out_dim = int(training_cfg.get("out_dim", 128))
    layers = int(training_cfg.get("layers", 2))
    dropout = float(training_cfg.get("dropout", 0.0))
    ablation = gnn_encoder_ablation_from_training_cfg(training_cfg)
    if ablation.enabled:
        model: PairGnnEncoder | HeteroSAGE = PairGnnEncoder(
            metadata=metadata,
            hidden=hidden,
            out=out_dim,
            layers=layers,
            dropout=dropout,
            ablation=ablation,
        )
    else:
        model = HeteroSAGE(metadata=metadata, hidden=hidden, out=out_dim, layers=layers, dropout=dropout)
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=True)
    if device is not None:
        model = model.to(device)
    return model


class DotPredictor(nn.Module):
    def forward(self, src, dst, edge_types=None):
        return (src * dst).sum(dim=-1)

class MLPredictor(nn.Module):
    def __init__(self, d, h=128, edge_types=None, dropout=0.3, use_dropout=True):
        super().__init__()
        layers = [
            nn.Linear(2 * d, h),
            nn.ReLU(),
        ]
        if use_dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(h, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, src, dst):
        return self.net(torch.cat([src, dst], dim=-1)).squeeze(-1)

class DistMultPredictor(nn.Module):
    def __init__(self, dim, edge_types):
        super().__init__()
        self.dim = dim
        self.rel_params = nn.ParameterDict()

        sigma = 1.0 / math.sqrt(dim)

        for et in edge_types:
            key = "__".join(et)
            param = nn.Parameter(torch.empty(dim))
            nn.init.normal_(param, mean=0.0, std=sigma)
            self.rel_params[key] = param

    def forward(self, src, dst, edge_type):
        key = "__".join(edge_type)
        r = self.rel_params[key]
        return (src * r * dst).sum(dim=-1)


