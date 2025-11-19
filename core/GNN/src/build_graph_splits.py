import math
import torch
from torch import nn

# -----------------------------
# 1) Pick relations to supervise
# -----------------------------
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


# -----------------------------
# 2) Train/Val/Test split per relation + build a leakage-safe training graph
# -----------------------------
def split_edges_and_build_train_graph(TORCH_SEED, data, edge_types, val_ratio=0.1, test_ratio=0.1):
    train_pos, val_pos, test_pos = {}, {}, {}
    train_graph = data.clone()

    g = torch.Generator().manual_seed(TORCH_SEED)

    #For each realation (e.g. ('movie, 'to' 'actor'))
    for et in edge_types:

        #Get full list of edges for that relation
        ei = data[et].edge_index

        #ei is the edge_index tensor of shape [2, E], (rows = source/destination; columns = individual edges)
        E = ei.size(1)

        #permute edge rows with a fixed seed
        perm = torch.randperm(E, generator=g)

        # Partitions edges into train / val / test (80/10/10)
        #Get no. of rows for val corresponding 10%
        n_val = int(math.floor(E * val_ratio))

        #Get no. of rows for test corresponding 10%
        n_test = int(math.floor(E * test_ratio))

        #Get all row indexes for val
        val_idx = perm[:n_val]

        #Get all row indexes for test
        test_idx = perm[n_val:n_val + n_test]

        #Get all remaining row indexes for training
        train_idx = perm[n_val + n_test:]

        # Get the rows for train, val and test
        train_pos[et] = ei[:, train_idx].contiguous()
        val_pos[et]   = ei[:, val_idx].contiguous()
        test_pos[et]  = ei[:, test_idx].contiguous()

        # IMPORTANT: Replace edges for the current edge type with only training edges in the train_graph.
        train_graph[et].edge_index = train_pos[et]

    return train_graph, train_pos, val_pos, test_pos