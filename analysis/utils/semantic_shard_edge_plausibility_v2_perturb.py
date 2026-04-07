"""
Simple perturbations for Method 1 V2: feature dropout, noise, optional channel dropout.

Does **not** recompute local graph structure.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def perturb_features(
    x: np.ndarray,
    rng: np.random.Generator,
    *,
    feature_names: list[str],
    manifest_groups: dict[str, list[str]],
    dropout_prob: float,
    noise_std: float,
    view_dropout_prob: float,
    use_view_dropout: bool,
) -> np.ndarray:
    """
    Apply multiplicative attenuation noise + elementwise dropout mask + optional whole-group dropout.

    ``x`` is (batch, n_features) aligned with ``feature_names``.
    """
    out = x.astype(np.float64).copy()
    n, d = out.shape
    if d != len(feature_names):
        raise ValueError("feature_names length must match x.shape[1]")

    name_to_idx = {nm: i for i, nm in enumerate(feature_names)}

    # Optional: drop entire semantic / infra / temporal **feature** groups (not local/hub/nodepair)
    if use_view_dropout and rng.random() < view_dropout_prob:
        groups = ["semantic", "infra", "temporal"]
        g = rng.choice(groups)
        for col in manifest_groups.get(g, []):
            j = name_to_idx.get(col)
            if j is not None:
                out[:, j] = 0.0

    # Elementwise dropout mask (zeroing)
    if dropout_prob > 0:
        mask = rng.random(out.shape) > dropout_prob
        out *= mask.astype(np.float64)

    # Small multiplicative lognormal-ish noise around 1
    if noise_std > 0:
        noise = rng.normal(1.0, noise_std, size=out.shape)
        noise = np.clip(noise, 0.0, 3.0)
        out *= noise

    return out
