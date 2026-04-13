"""
Separate projections for email node features to balance dimensionality.

Raw email features are laid out as:
  [scalars (4), SBERT(subject), SBERT(body), html_css (40), bool_attrs (7), auth_onehot (18)]
  = [ts, len_body, n_urls, len_subject, subj_emb, body_emb, html_css, bools, spf/dkim/dmarc one-hot]

BERT embeddings dominate the raw feature space (e.g. 1024 dims each). This module
projects only the BERT block down to the same width as the structured block (50/50),
and passes scalars + html_css + bools + auth_onehot through unchanged (no learned
down-projection on structured attributes).

Usage:
  from graph.feature_projection import EmailFeatureProjection, email_feature_layout

  proj = EmailFeatureProjection(subj_dim=1024, body_dim=1024)
  data["email"].x = proj(data["email"].x)  # e.g. after loading graph
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .common import AUTH_ONEHOT_DIM

# Layout of raw email.x (must match assembler and extract_bert_embeddings)
SCALAR_COUNT = 4  # ts, len_body, n_urls, len_subject
HTML_CSS_LEN = 40  # len(create_html_css_features({}, {}))
BOOL_ATTR_COUNT = 7  # cyrillic_domain, contains_symbols, body_has_tracking_*, etc.


def structured_other_in_dim(html_css_len: int) -> int:
    """Width of the non-BERT structured block: scalars + html/css + bools + auth one-hot."""
    return SCALAR_COUNT + html_css_len + BOOL_ATTR_COUNT + AUTH_ONEHOT_DIM


# Non-BERT feature dim when HTML/CSS features are included (default graph build)
OTHER_FEATURE_DIM = structured_other_in_dim(HTML_CSS_LEN)

# After projection with defaults (both *_out_dim null): BERT -> OTHER_FEATURE_DIM + structured passthrough
PROJECTED_EMAIL_FEATURE_DIM = 2 * OTHER_FEATURE_DIM


@dataclass(frozen=True)
class ResolvedEmailProjectionDims:
    """Effective output widths and whether a linear maps the structured block."""

    bert_out: int
    other_out: int
    project_other: bool


def resolve_email_projection_dims(
    other_in_dim: int,
    *,
    bert_out_dim: int | None,
    other_out_dim: int | None,
) -> ResolvedEmailProjectionDims:
    """
    Pipeline / CLI resolution for ``bert_out_dim`` and ``other_out_dim``.

    - ``other_out_dim`` None: keep structured features at full width (no linear).
    - ``other_out_dim`` set: ``Linear(other_in_dim → other_out_dim)``.
    - ``bert_out_dim`` None: SBERT maps to ``other_out`` so channel counts match (50/50 when both sides use the same width).
    - ``bert_out_dim`` set: SBERT maps to that width.
    """
    if other_out_dim is None:
        other_eff = other_in_dim
        project_other = False
    else:
        if other_out_dim <= 0:
            raise ValueError("other_out_dim must be positive when set.")
        other_eff = other_out_dim
        project_other = True

    if bert_out_dim is None:
        bert_eff = other_eff
    else:
        if bert_out_dim <= 0:
            raise ValueError("bert_out_dim must be positive when set.")
        bert_eff = bert_out_dim

    return ResolvedEmailProjectionDims(
        bert_out=bert_eff,
        other_out=other_eff,
        project_other=project_other,
    )


def email_feature_layout(
    subj_dim: int,
    body_dim: int,
    *,
    html_css_len: int = HTML_CSS_LEN,
) -> Tuple[int, int, int]:
    """Return (bert_dim, other_dim, total_raw_dim) for the given embedding dims."""
    bert_dim = subj_dim + body_dim
    other_dim = structured_other_in_dim(html_css_len)
    total = SCALAR_COUNT + bert_dim + html_css_len + BOOL_ATTR_COUNT + AUTH_ONEHOT_DIM
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
    SBERT block → linear; structured block → optional linear or passthrough; concat.
    Defaults (both ``*_out_dim`` unset): BERT to structured width + full structured vector (50/50 channels).
    """

    def __init__(
        self,
        subj_dim: int,
        body_dim: int,
        *,
        html_css_len: int = HTML_CSS_LEN,
        bert_out_dim: int | None = None,
        other_out_dim: int | None = None,
    ):
        if subj_dim < 0 or body_dim < 0:
            raise ValueError("subj_dim and body_dim must be non-negative")
        self.subj_dim = subj_dim
        self.body_dim = body_dim
        self.bert_in_dim = subj_dim + body_dim
        self.html_css_len = html_css_len
        self.other_in_dim = SCALAR_COUNT + html_css_len + BOOL_ATTR_COUNT + AUTH_ONEHOT_DIM

        resolved = resolve_email_projection_dims(
            self.other_in_dim,
            bert_out_dim=bert_out_dim,
            other_out_dim=other_out_dim,
        )
        self._resolved = resolved

        nn = _get_nn()
        if self.bert_in_dim > 0:
            self.bert_proj = nn.Linear(self.bert_in_dim, resolved.bert_out)
        else:
            self.bert_proj = None
        if resolved.project_other:
            self.other_proj = nn.Linear(self.other_in_dim, resolved.other_out)
        else:
            self.other_proj = None
        self._out_dim = (
            (resolved.bert_out if self.bert_in_dim > 0 else 0) + resolved.other_out
        )

    @property
    def out_dim(self) -> int:
        """Output feature dimension after projection."""
        return self._out_dim

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Project raw email features [N, raw_dim] to [N, out_dim]."""
        torch = _get_torch()
        if x.dim() != 2:
            raise ValueError("Expected 2D tensor [N, raw_dim]")
        # Other part: scalars [0:4], html_css [4+bert_dim : 4+bert_dim+40], bools+auth_onehot [-7-18:]
        start_html = SCALAR_COUNT + self.bert_in_dim
        end_html = start_html + self.html_css_len
        trail_len = BOOL_ATTR_COUNT + AUTH_ONEHOT_DIM
        other_parts = [
            x[:, :SCALAR_COUNT],
            x[:, start_html:end_html],
            x[:, -trail_len:],
        ]
        other_cat = torch.cat(other_parts, dim=1)
        if self.other_proj is not None:
            other_out = self.other_proj(other_cat)
        else:
            other_out = other_cat
        if self.bert_proj is not None:
            bert_part = x[:, SCALAR_COUNT : SCALAR_COUNT + self.bert_in_dim]
            bert_out = self.bert_proj(bert_part)
            return torch.cat([bert_out, other_out], dim=1)
        return other_out

    def __call__(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.forward(x)


class EmailFeatureProjectionModule(_ProjectionBase):
    """
    nn.Module version of EmailFeatureProjection for use in PyTorch graph builds.
    """

    def __init__(
        self,
        subj_dim: int,
        body_dim: int,
        *,
        html_css_len: int = HTML_CSS_LEN,
        bert_out_dim: int | None = None,
        other_out_dim: int | None = None,
    ):
        super().__init__()
        if subj_dim < 0 or body_dim < 0:
            raise ValueError("subj_dim and body_dim must be non-negative")
        self.subj_dim = subj_dim
        self.body_dim = body_dim
        self.bert_in_dim = subj_dim + body_dim
        self.html_css_len = html_css_len
        self.other_in_dim = SCALAR_COUNT + html_css_len + BOOL_ATTR_COUNT + AUTH_ONEHOT_DIM

        resolved = resolve_email_projection_dims(
            self.other_in_dim,
            bert_out_dim=bert_out_dim,
            other_out_dim=other_out_dim,
        )
        self._resolved = resolved

        nn = _get_nn()
        if self.bert_in_dim > 0:
            self.bert_proj = nn.Linear(self.bert_in_dim, resolved.bert_out)
        else:
            self.bert_proj = None
        if resolved.project_other:
            self.other_proj = nn.Linear(self.other_in_dim, resolved.other_out)
        else:
            self.other_proj = None
        self._out_dim = (
            (resolved.bert_out if self.bert_in_dim > 0 else 0) + resolved.other_out
        )

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        torch = _get_torch()
        if x.dim() != 2:
            raise ValueError("Expected 2D tensor [N, raw_dim]")
        start_html = SCALAR_COUNT + self.bert_in_dim
        end_html = start_html + self.html_css_len
        trail_len = BOOL_ATTR_COUNT + AUTH_ONEHOT_DIM
        other_parts = [
            x[:, :SCALAR_COUNT],
            x[:, start_html:end_html],
            x[:, -trail_len:],
        ]
        other_cat = torch.cat(other_parts, dim=1)
        if self.other_proj is not None:
            other_out = self.other_proj(other_cat)
        else:
            other_out = other_cat
        if self.bert_proj is not None:
            bert_part = x[:, SCALAR_COUNT : SCALAR_COUNT + self.bert_in_dim]
            bert_out = self.bert_proj(bert_part)
            return torch.cat([bert_out, other_out], dim=1)
        return other_out


__all__ = [
    "SCALAR_COUNT",
    "HTML_CSS_LEN",
    "BOOL_ATTR_COUNT",
    "AUTH_ONEHOT_DIM",
    "structured_other_in_dim",
    "OTHER_FEATURE_DIM",
    "PROJECTED_EMAIL_FEATURE_DIM",
    "ResolvedEmailProjectionDims",
    "resolve_email_projection_dims",
    "email_feature_layout",
    "EmailFeatureProjection",
    "EmailFeatureProjectionModule",
]
