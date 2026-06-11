import torch

def _replace_outliers_with_median(
    x: torch.Tensor,
    *,
    iqr_multiplier: float = 1.5,
    min_rows_for_outlier_detection: int = 8,
) -> torch.Tensor:
    """
    Robustly clean outliers per feature column.

    Outliers are detected with IQR bounds and replaced by the column median.
    This prevents extreme values from dominating normalization statistics while
    preserving the number of rows/nodes (so graph connectivity stays valid).
    """
    if x.numel() == 0:
        return x

    cleaned = x.clone()
    n_rows, n_cols = cleaned.shape
    if n_rows < min_rows_for_outlier_detection:
        return cleaned

    for col_idx in range(n_cols):
        col = cleaned[:, col_idx]
        finite_mask = torch.isfinite(col)
        if not torch.any(finite_mask):
            cleaned[:, col_idx] = 0.0
            continue

        finite_vals = col[finite_mask]
        if finite_vals.numel() < min_rows_for_outlier_detection:
            continue

        q1 = torch.quantile(finite_vals, 0.25)
        q3 = torch.quantile(finite_vals, 0.75)
        iqr = q3 - q1
        median = torch.median(finite_vals)

        if torch.abs(iqr) < 1e-12:
            cleaned[~finite_mask, col_idx] = median
            continue

        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr

        outlier_mask = finite_mask & ((col < lower) | (col > upper))
        cleaned[outlier_mask, col_idx] = median
        cleaned[~finite_mask, col_idx] = median

    return cleaned


def normalize_graph(
    data,
    *,
    iqr_multiplier: float = 1.5,
    min_rows_for_outlier_detection: int = 8,
):
    """
    Robust standardization for all node feature matrices in a HeteroData graph.

    Steps per node type:
    1) Detect outliers per feature dim via IQR rule.
    2) Replace outliers with per-dim median.
    3) Standardize to zero mean and unit variance.
    """
    for ntype in data.node_types:
        if "x" not in data[ntype]:
            continue

        x = data[ntype].x
        if x.dtype != torch.float32:
            x = x.float()

        x_clean = _replace_outliers_with_median(
            x,
            iqr_multiplier=iqr_multiplier,
            min_rows_for_outlier_detection=min_rows_for_outlier_detection,
        )

        mu = x_clean.mean(dim=0)
        sigma = x_clean.std(dim=0).clamp_min(1e-6)
        data[ntype].x = (x_clean - mu) / sigma

    return data


