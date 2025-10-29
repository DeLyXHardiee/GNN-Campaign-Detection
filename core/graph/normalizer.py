from typing import List, Tuple
import math

def zscore_list(values: List[float]) -> Tuple[List[float], float, float]:
    """Compute z-score normalization for a 1D list.

    Returns (normalized_values, mean, std). If std is 0 or list empty, returns zeros.
    """
    n = len(values)
    if n == 0:
        return [], 0.0, 0.0
    mean = sum(values) / float(n)
    # population std (can switch to sample if desired)
    var = 0.0
    for v in values:
        dv = v - mean
        var += dv * dv
    var /= float(n)
    std = math.sqrt(var)
    if std <= 1e-12:
        return [0.0 for _ in values], mean, 0.0
    return [float((v - mean) / std) for v in values], mean, std


def minmax_list(values: List[float]) -> Tuple[List[float], float, float]:
    """Compute min-max scaling to [0,1] for a 1D list.

    Returns (normalized_values, vmin, vmax). If vmax==vmin or empty, returns zeros.
    """
    if not values:
        return [], 0.0, 0.0
    vmin = min(values)
    vmax = max(values)
    rng = float(vmax - vmin)
    if rng <= 0.0:
        return [0.0 for _ in values], float(vmin), float(vmax)
    return [float((v - vmin) / rng) for v in values], float(vmin), float(vmax)