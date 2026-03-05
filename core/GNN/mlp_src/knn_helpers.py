import numpy as np
from typing import Tuple

def show_topk_cosine_neighbors(
    bodies: list[str],
    embeddings: np.ndarray,
    query_idx: int,
    k: int = 10,
    exclude_self: bool = True,
    max_chars: int = 400,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes top-k cosine neighbors for `query_idx` and prints the query + neighbors.

    Assumes embeddings are L2-normalized so cosine similarity == dot product.

    Returns:
      neighbor_indices: (k,)
      neighbor_sims:    (k,)
    """
    n = embeddings.shape[0]
    if len(bodies) != n:
        raise ValueError(f"Mismatch: {len(bodies)} bodies vs {n} embeddings rows")
    if not (0 <= query_idx < n):
        raise IndexError(f"query_idx {query_idx} out of range [0, {n-1}]")

    q = embeddings[query_idx]      # (D,)
    sims = embeddings @ q          # (N,)

    if exclude_self:
        sims[query_idx] = -np.inf

    k = min(k, n - (1 if exclude_self else 0))
    cand = np.argpartition(-sims, k)[:k]
    cand = cand[np.argsort(-sims[cand])]

    neighbor_sims = sims[cand]

    # Print
    print(f"\n=== QUERY idx={query_idx} ===")
    print(bodies[query_idx][:max_chars])

    print("\n=== NEIGHBORS ===")
    for rank, (j, s) in enumerate(zip(cand.tolist(), neighbor_sims.tolist()), start=1):
        print(f"\n--- #{rank} idx={j}  cosine_sim={s:.4f} ---")
        print(bodies[j][:max_chars])

    return cand, neighbor_sims


