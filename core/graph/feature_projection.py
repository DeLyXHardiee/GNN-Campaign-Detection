"""
Separate projections for email node features to balance dimensionality.

Raw email features are laid out as:
  [scalars (4), SBERT(subject), SBERT(body), html_css (40), bool_attrs (7)]
  = [ts, len_body, n_urls, len_subject, subj_emb, body_emb, html_css, bools]

BERT embeddings dominate the feature space (e.g. 1024 dims each). This module
projects BERT down and projects the rest (scalars + html_css + bools) up,
then concatenates for a balanced email feature vector.

Usage:
  from graph.feature_projection import EmailFeatureProjection, email_feature_layout

  proj = EmailFeatureProjection(subj_dim=1024, body_dim=1024)
  data["email"].x = proj(data["email"].x)  # e.g. after loading graph
"""
from __future__ import annotations

from typing import Tuple

# Layout of raw email.x (must match assembler and extract_bert_embeddings)
SCALAR_COUNT = 4  # ts, len_body, n_urls, len_subject
HTML_CSS_LEN = 40  # len(create_html_css_features({}, {}))
BOOL_ATTR_COUNT = 7  # cyrillic_domain, contains_symbols, body_has_tracking_*, etc.

# Non-BERT feature dim: scalars + html_css + bools
OTHER_FEATURE_DIM = SCALAR_COUNT + HTML_CSS_LEN + BOOL_ATTR_COUNT  # 51


def email_feature_layout(
    subj_dim: int,
    body_dim: int,
) -> Tuple[int, int, int]:
    """Return (bert_dim, other_dim, total_raw_dim) for the given embedding dims."""
    bert_dim = subj_dim + body_dim
    other_dim = OTHER_FEATURE_DIM
    total = SCALAR_COUNT + bert_dim + HTML_CSS_LEN + BOOL_ATTR_COUNT
    return bert_dim, other_dim, total


def _get_torch():
    try:
        import torch
        return torch
    except ImportError as e:
        raise ImportError(
            "feature_projection requires PyTorch. Install with: pip install torch"
        ) from e


def _get_nn():
    _get_torch()
    from torch import nn
    return nn


try:
    from torch import nn as _NN
    _ProjectionBase = _NN.Module
except ImportError:
    _NN = None
    _ProjectionBase = object


class EmailFeatureProjection:
    """
    Projects raw email node features into a balanced space:
      - BERT part (subject + body embeddings) → linear down to bert_out_dim (default 128)
      - Other part (scalars + html_css + bools) → linear up to other_out_dim (default 32)
      - Output = concat(bert_proj, other_proj)
    """

    def __init__(
        self,
        subj_dim: int,
        body_dim: int,
        *,
        bert_out_dim: int = 128,
        other_out_dim: int = 32,
        html_css_len: int = HTML_CSS_LEN,
    ):
        if subj_dim < 0 or body_dim < 0:
            raise ValueError("subj_dim and body_dim must be non-negative")
        self.subj_dim = subj_dim
        self.body_dim = body_dim
        self.bert_in_dim = subj_dim + body_dim
        self.bert_out_dim = bert_out_dim
        self.other_out_dim = other_out_dim
        self.html_css_len = html_css_len
        self.other_in_dim = SCALAR_COUNT + html_css_len + BOOL_ATTR_COUNT

        nn = _get_nn()
        if self.bert_in_dim > 0:
            self.bert_proj = nn.Linear(self.bert_in_dim, bert_out_dim)
        else:
            self.bert_proj = None
        self.other_proj = nn.Linear(self.other_in_dim, other_out_dim)
        self._out_dim = (bert_out_dim if self.bert_in_dim > 0 else 0) + other_out_dim

    @property
    def out_dim(self) -> int:
        """Output feature dimension after projection."""
        return self._out_dim

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Project raw email features [N, raw_dim] to [N, out_dim]."""
        torch = _get_torch()
        if x.dim() != 2:
            raise ValueError("Expected 2D tensor [N, raw_dim]")
        # Other part: scalars [0:4], html_css [4+bert_dim : 4+bert_dim+40], bools [-7:]
        start_html = SCALAR_COUNT + self.bert_in_dim
        end_html = start_html + self.html_css_len
        other_parts = [
            x[:, :SCALAR_COUNT],
            x[:, start_html:end_html],
            x[:, -BOOL_ATTR_COUNT:],
        ]
        other = torch.cat(other_parts, dim=1)
        other_proj = self.other_proj(other)
        if self.bert_proj is not None:
            bert_part = x[:, SCALAR_COUNT : SCALAR_COUNT + self.bert_in_dim]
            bert_proj = self.bert_proj(bert_part)
            return torch.cat([bert_proj, other_proj], dim=1)
        return other_proj

    def __call__(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.forward(x)


class EmailFeatureProjectionModule(_ProjectionBase):
    """
    nn.Module version of EmailFeatureProjection for use in PyTorch models.
    Use this when the projection should be a trainable part of the model.
    """

    def __init__(
        self,
        subj_dim: int,
        body_dim: int,
        *,
        bert_out_dim: int = 128,
        other_out_dim: int = 32,
        html_css_len: int = HTML_CSS_LEN,
    ):
        super().__init__()
        if subj_dim < 0 or body_dim < 0:
            raise ValueError("subj_dim and body_dim must be non-negative")
        self.subj_dim = subj_dim
        self.body_dim = body_dim
        self.bert_in_dim = subj_dim + body_dim
        self.bert_out_dim = bert_out_dim
        self.other_out_dim = other_out_dim
        self.html_css_len = html_css_len
        self.other_in_dim = SCALAR_COUNT + html_css_len + BOOL_ATTR_COUNT

        nn = _get_nn()
        if self.bert_in_dim > 0:
            self.bert_proj = nn.Linear(self.bert_in_dim, bert_out_dim)
        else:
            self.bert_proj = None
        self.other_proj = nn.Linear(self.other_in_dim, other_out_dim)
        self._out_dim = (bert_out_dim if self.bert_in_dim > 0 else 0) + other_out_dim

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        torch = _get_torch()
        if x.dim() != 2:
            raise ValueError("Expected 2D tensor [N, raw_dim]")
        start_html = SCALAR_COUNT + self.bert_in_dim
        end_html = start_html + self.html_css_len
        other_parts = [
            x[:, :SCALAR_COUNT],
            x[:, start_html:end_html],
            x[:, -BOOL_ATTR_COUNT:],
        ]
        other = torch.cat(other_parts, dim=1)
        other_proj = self.other_proj(other)
        if self.bert_proj is not None:
            bert_part = x[:, SCALAR_COUNT : SCALAR_COUNT + self.bert_in_dim]
            bert_proj = self.bert_proj(bert_part)
            return torch.cat([bert_proj, other_proj], dim=1)
        return other_proj


__all__ = [
    "SCALAR_COUNT",
    "HTML_CSS_LEN",
    "BOOL_ATTR_COUNT",
    "OTHER_FEATURE_DIM",
    "email_feature_layout",
    "EmailFeatureProjection",
    "EmailFeatureProjectionModule",
]
