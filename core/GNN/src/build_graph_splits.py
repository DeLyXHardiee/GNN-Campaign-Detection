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


def _find_reverse_et(et, supervised_set):
    """Return the reverse edge type (dst, rev_rel, src) if it exists in supervised_set."""
    src, rel, dst = et
    rev = (dst, f"rev_{rel}", src)
    return rev if rev in supervised_set else None


def split_edges_coordinated(TORCH_SEED, data, edge_types, val_ratio=0.1, test_ratio=0.1):
    """
    Like split_edges_and_build_train_graph but forward/reverse edge type pairs share
    the same random permutation, so test[forward_i] always corresponds to test[reverse_i].
    This prevents the GNN from seeing withheld test edges via their reverse direction.

    Assumes ToUndirected() was applied: edge k in (A,rel,B) corresponds to edge k in
    (B,rev_rel,A). Unpaired edge types are split independently as usual.
    """
    train_pos, val_pos, test_pos = {}, {}, {}
    train_graph = data.clone()
    g = torch.Generator().manual_seed(TORCH_SEED)
    supervised_set = {tuple(et) for et in edge_types}
    done = set()

    for et in edge_types:
        et = tuple(et)
        if et in done:
            continue
        rev_et = _find_reverse_et(et, supervised_set)

        ei_fwd = data[et].edge_index
        E = ei_fwd.size(1)
        perm = torch.randperm(E, generator=g)
        n_val = int(math.floor(E * val_ratio))
        n_test = int(math.floor(E * test_ratio))
        val_idx = perm[:n_val]
        test_idx = perm[n_val:n_val + n_test]
        train_idx = perm[n_val + n_test:]

        pairs = [(et, ei_fwd)]
        if rev_et is not None:
            pairs.append((rev_et, data[rev_et].edge_index))

        for _et, _ei in pairs:
            train_pos[_et] = _ei[:, train_idx].contiguous()
            val_pos[_et] = _ei[:, val_idx].contiguous()
            test_pos[_et] = _ei[:, test_idx].contiguous()
            train_graph[_et].edge_index = train_pos[_et]
            done.add(_et)

    return train_graph, train_pos, val_pos, test_pos


def split_email_nodes_inductively(TORCH_SEED, data, edge_types, val_ratio=0.1, test_ratio=0.1):
    """
    Hold out val_ratio/test_ratio of EMAIL NODES entirely.
    All edges incident to held-out emails are withheld from training.
    The GNN must produce embeddings for test emails using only their node features
    (no neighbour aggregation), testing true inductive generalisation.
    """
    n_emails = data['email'].num_nodes
    g = torch.Generator().manual_seed(TORCH_SEED)
    perm = torch.randperm(n_emails, generator=g)
    n_val = int(math.floor(n_emails * val_ratio))
    n_test = int(math.floor(n_emails * test_ratio))

    val_email_mask = torch.zeros(n_emails, dtype=torch.bool)
    test_email_mask = torch.zeros(n_emails, dtype=torch.bool)
    val_email_mask[perm[:n_val]] = True
    test_email_mask[perm[n_val:n_val + n_test]] = True

    train_pos, val_pos, test_pos = {}, {}, {}
    train_graph = data.clone()

    for et in edge_types:
        src_type, _, dst_type = tuple(et)
        ei = data[et].edge_index          

        if src_type == 'email':
            email_row = ei[0]
        elif dst_type == 'email':
            email_row = ei[1]
        else:
            E = ei.size(1)
            perm2 = torch.randperm(E, generator=g)
            n_v = int(math.floor(E * val_ratio))
            n_t = int(math.floor(E * test_ratio))
            train_pos[et] = ei[:, perm2[n_v + n_t:]].contiguous()
            val_pos[et] = ei[:, perm2[:n_v]].contiguous()
            test_pos[et] = ei[:, perm2[n_v:n_v + n_t]].contiguous()
            train_graph[et].edge_index = train_pos[et]
            continue

        is_test = test_email_mask[email_row]
        is_val = val_email_mask[email_row]
        is_train = ~is_test & ~is_val

        train_pos[et] = ei[:, is_train].contiguous()
        val_pos[et] = ei[:, is_val].contiguous()
        test_pos[et] = ei[:, is_test].contiguous()
        train_graph[et].edge_index = train_pos[et]

    return train_graph, train_pos, val_pos, test_pos