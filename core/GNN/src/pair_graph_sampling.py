"""
Substep 3: explicit hetero neighborhood sampling around email pair endpoints.

Contract: seed unique global email indices from the pair batch, run NeighborLoader
on the full hetero evidence graph (all node/edge types), then expose global→local
email mapping and per-pair local indices for encoder output h[\"email\"].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch_geometric.loader import NeighborLoader


@dataclass
class PairEndpointCoverageDiag:
    """Per (pair-chunk, sampled-subgraph) coverage statistics."""

    n_pairs_requested: int
    n_both_endpoints_present: int
    n_missing_i_only: int
    n_missing_j_only: int
    n_missing_both_endpoints: int
    frac_usable_pairs: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_pairs_requested": self.n_pairs_requested,
            "n_both_endpoints_present": self.n_both_endpoints_present,
            "n_missing_i_only": self.n_missing_i_only,
            "n_missing_j_only": self.n_missing_j_only,
            "n_missing_both_endpoints": self.n_missing_both_endpoints,
            "frac_usable_pairs": self.frac_usable_pairs,
        }


@dataclass
class PairEndpointHeteroSample:
    """
    One sampled hetero minibatch anchored on unique email endpoints from a pair row chunk.

    Fields are CPU tensors / CPU hetero batch unless noted; move ``hetero_batch`` to device
    before the encoder forward.
    """

    hetero_batch: Any
    pair_local_i: torch.Tensor
    """LongTensor (B,), local email row in batch h[\"email\"]; -1 if endpoint i absent."""
    pair_local_j: torch.Tensor
    pair_ok_mask: torch.Tensor
    """Bool (B,): both endpoints present in sampled email n_id set."""
    global_to_local_email: dict[int, int]
    coverage: PairEndpointCoverageDiag
    seed_global_email_indices: torch.Tensor
    """LongTensor (S,) unique seeds passed to NeighborLoader."""
    hetero_total_nodes: int
    hetero_total_edges: int
    n_email_nodes_in_batch: int
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _num_neighbors_dict(data: Any, fanout: list[int]) -> dict[tuple[str, str, str], list[int]]:
    return {et: list(fanout) for et in data.edge_types}


def _global_email_to_local_map(batch_email_store: Any) -> dict[int, int]:
    n_id = batch_email_store.n_id
    if n_id.dim() != 1:
        raise ValueError("Expected 1D email n_id on batch.")
    out: dict[int, int] = {}
    for loc in range(int(n_id.numel())):
        gid = int(n_id[loc].item())
        out[gid] = loc
    return out


def _hetero_batch_total_nodes(hetero_batch: Any) -> int:
    return int(sum(int(hetero_batch[nt].num_nodes) for nt in hetero_batch.node_types))


def _hetero_batch_total_edges(hetero_batch: Any) -> int:
    return int(sum(int(hetero_batch[et].edge_index.size(1)) for et in hetero_batch.edge_types))


def _pair_endpoint_coverage(gmap: dict[int, int], gi: np.ndarray, gj: np.ndarray) -> PairEndpointCoverageDiag:
    n_req = int(len(gi))
    n_both = n_miss_i = n_miss_j = n_miss_both = 0
    for a, b in zip(gi.tolist(), gj.tolist(), strict=False):
        ha = int(a) in gmap
        hb = int(b) in gmap
        if ha and hb:
            n_both += 1
        elif not ha and not hb:
            n_miss_both += 1
        elif not ha:
            n_miss_i += 1
        else:
            n_miss_j += 1
    frac = float(n_both / max(n_req, 1))
    return PairEndpointCoverageDiag(
        n_pairs_requested=n_req,
        n_both_endpoints_present=n_both,
        n_missing_i_only=n_miss_i,
        n_missing_j_only=n_miss_j,
        n_missing_both_endpoints=n_miss_both,
        frac_usable_pairs=frac,
    )


def sample_hetero_around_pair_endpoints(
    data_cpu: Any,
    gi: np.ndarray,
    gj: np.ndarray,
    fanout: list[int],
) -> PairEndpointHeteroSample:
    """
    Sample a hetero subgraph with NeighborLoader seeded at all unique graph email indices
    appearing in (gi, gj). Message passing remains on the full hetero schema.
    """
    uniq = np.unique(np.concatenate([gi, gj])).astype(np.int64)
    if uniq.size == 0:
        raise ValueError("sample_hetero_around_pair_endpoints: empty endpoint set.")
    seed = torch.as_tensor(uniq, dtype=torch.long)
    num_neighbors = _num_neighbors_dict(data_cpu, fanout)
    loader = NeighborLoader(
        data_cpu,
        num_neighbors=num_neighbors,
        input_nodes=("email", seed),
        batch_size=int(seed.numel()),
        shuffle=False,
        num_workers=0,
    )
    hetero_batch = next(iter(loader))
    gmap = _global_email_to_local_map(hetero_batch["email"])
    cov = _pair_endpoint_coverage(gmap, gi, gj)

    loc_i: list[int] = []
    loc_j: list[int] = []
    ok: list[bool] = []
    for a, b in zip(gi.tolist(), gj.tolist(), strict=False):
        ii = gmap.get(int(a))
        jj = gmap.get(int(b))
        present = ii is not None and jj is not None
        ok.append(present)
        loc_i.append(int(ii) if ii is not None else -1)
        loc_j.append(int(jj) if jj is not None else -1)

    pair_local_i = torch.tensor(loc_i, dtype=torch.long)
    pair_local_j = torch.tensor(loc_j, dtype=torch.long)
    pair_ok_mask = torch.tensor(ok, dtype=torch.bool)
    n_email = int(hetero_batch["email"].num_nodes)
    diag = {
        "n_unique_seed_emails": int(uniq.size),
        "n_email_nodes_in_sampled_batch": n_email,
        "hetero_total_nodes": _hetero_batch_total_nodes(hetero_batch),
        "hetero_total_edges": _hetero_batch_total_edges(hetero_batch),
        "coverage": cov.to_dict(),
    }
    return PairEndpointHeteroSample(
        hetero_batch=hetero_batch,
        pair_local_i=pair_local_i,
        pair_local_j=pair_local_j,
        pair_ok_mask=pair_ok_mask,
        global_to_local_email=gmap,
        coverage=cov,
        seed_global_email_indices=seed,
        hetero_total_nodes=diag["hetero_total_nodes"],
        hetero_total_edges=diag["hetero_total_edges"],
        n_email_nodes_in_batch=n_email,
        diagnostics=diag,
    )


def collect_pair_sampling_diagnostics(
    data_cpu: Any,
    df: Any,
    *,
    pair_batch_iter: Any,
    pair_batch_size: int,
    max_unique_emails: int,
    fanout: list[int],
    max_batches: int | None = 200,
    assert_full_endpoint_coverage: bool = False,
) -> dict[str, Any]:
    """
    Structural validation over pair batches (no encoder). Optionally assert every pair maps.

    ``pair_batch_iter`` must match training (e.g. ``iter_pair_batches`` from ``pair_train``).
    """
    uniq_seeds: list[int] = []
    tot_nodes: list[int] = []
    tot_edges: list[int] = []
    n_email_nodes: list[int] = []
    frac_ok: list[float] = []
    n_miss_i: list[int] = []
    n_miss_j: list[int] = []
    n_miss_both: list[int] = []
    catastrophic = False
    n_seen = 0

    for _chunk, gi, gj in pair_batch_iter(df, pair_batch_size, max_unique_emails):
        sample = sample_hetero_around_pair_endpoints(data_cpu, gi, gj, fanout)
        c = sample.coverage
        uniq_seeds.append(int(sample.seed_global_email_indices.numel()))
        tot_nodes.append(sample.hetero_total_nodes)
        tot_edges.append(sample.hetero_total_edges)
        n_email_nodes.append(sample.n_email_nodes_in_batch)
        frac_ok.append(c.frac_usable_pairs)
        n_miss_i.append(c.n_missing_i_only)
        n_miss_j.append(c.n_missing_j_only)
        n_miss_both.append(c.n_missing_both_endpoints)
        if c.frac_usable_pairs == 0.0 and c.n_pairs_requested > 0:
            catastrophic = True
        n_seen += 1
        if max_batches is not None and n_seen >= int(max_batches):
            break

    if n_seen == 0:
        return {
            "n_batches_observed": 0,
            "note": "no batches (empty dataframe or filters)",
        }

    def _mean(xs: list[float | int]) -> float:
        return float(sum(xs) / len(xs))

    min_f = float(min(frac_ok))
    out = {
        "n_batches_observed": n_seen,
        "avg_unique_seed_emails_per_graph_batch": _mean(uniq_seeds),
        "avg_sampled_hetero_total_nodes": _mean(tot_nodes),
        "avg_sampled_hetero_total_edges": _mean(tot_edges),
        "avg_n_email_nodes_in_sampled_batch": _mean(n_email_nodes),
        "avg_recoverable_pair_fraction": _mean(frac_ok),
        "min_recoverable_pair_fraction": min_f,
        "max_recoverable_pair_fraction": float(max(frac_ok)),
        "any_batch_catastrophic_endpoint_failure": catastrophic,
        "total_pairs_missing_i_endpoint_across_batches": int(sum(n_miss_i)),
        "total_pairs_missing_j_endpoint_across_batches": int(sum(n_miss_j)),
        "total_pairs_missing_both_endpoints_across_batches": int(sum(n_miss_both)),
    }
    if assert_full_endpoint_coverage and min_f < 1.0:
        raise AssertionError(
            f"pair_assert_full_endpoint_coverage: min recoverable fraction {min_f} < 1.0 "
            f"(see sampling_diagnostics)."
        )
    return out
