import math
import torch
from torch import nn

def pick_supervised_edge_types(data, primary_ntype='movie', direction='out'):
    """
    Return edge types to supervise that involve the primary node type.
    direction: 'out' | 'in' | 'both'
    """
    keep = []
    for et in data.edge_types:
        src, rel, dst = et
        if direction == 'out' and src == primary_ntype and dst != primary_ntype:
            keep.append(et)
        elif direction == 'in' and dst == primary_ntype and src != primary_ntype:
            keep.append(et)
        elif direction == 'both' and (src == primary_ntype or dst == primary_ntype):
            keep.append(et)
    keep = list(dict.fromkeys(keep))
    if not keep:
        raise ValueError(f"No edge types involving '{primary_ntype}' found. "
                         f"Available: {data.edge_types}")
    return keep


def split_edges_and_build_train_graph(TORCH_SEED, data, edge_types, val_ratio=0.1, test_ratio=0.1):
    train_pos, val_pos, test_pos = {}, {}, {}
    train_graph = data.clone()

    g = torch.Generator().manual_seed(TORCH_SEED)

    for et in edge_types:

        ei = data[et].edge_index

        E = ei.size(1)

        perm = torch.randperm(E, generator=g)

        n_val = int(math.floor(E * val_ratio))

        n_test = int(math.floor(E * test_ratio))

        val_idx = perm[:n_val]

        test_idx = perm[n_val:n_val + n_test]

        train_idx = perm[n_val + n_test:]

        train_pos[et] = ei[:, train_idx].contiguous()
        val_pos[et]   = ei[:, val_idx].contiguous()
        test_pos[et]  = ei[:, test_idx].contiguous()

        train_graph[et].edge_index = train_pos[et]

    return train_graph, train_pos, val_pos, test_pos