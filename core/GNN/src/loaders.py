import torch
try:
    import torch_sparse  # noqa: F401
except (ImportError, OSError):
    try:
        import pyg_lib  # noqa: F401
    except (ImportError, OSError):
        raise ImportError(
            "'NeighborSampler' requires either 'pyg-lib' or 'torch-sparse'. "
            "Install with: pip install torch-sparse (see PyG installation docs for your PyTorch/CUDA version)."
        ) from None

from torch_geometric.loader import LinkNeighborLoader, NeighborLoader

def make_link_loaders(train_graph, full_graph, train_pos, val_pos, test_pos,
                      edge_types, neg_ratio=1.0, batch_size=2048, fanout=[15, 10]):
    """
    Build LinkNeighborLoader for train/val/test. Data and edge_label_index are kept on CPU
    to avoid segmentation faults in PyG's NeighborSampler (see pyg-team/pytorch_geometric#7663).
    """
    loaders = {}
    # Ensure graph is on CPU; loaders expect CPU data.
    train_graph = train_graph.cpu() if hasattr(train_graph, "cpu") else train_graph

    num_neighbors = {et: fanout for et in full_graph.edge_types}

    for split_name, pos_dict in [('train', train_pos), ('val', val_pos), ('test', test_pos)]:
        split_loaders = {}

        for et in edge_types:
            pos_ei = pos_dict[et]
            if pos_ei.is_cuda:
                pos_ei = pos_ei.cpu()
            n = pos_ei.size(1)
            loader = LinkNeighborLoader(
                data=train_graph,
                num_neighbors=num_neighbors,
                edge_label_index=(et, pos_ei),
                edge_label=torch.ones(n, dtype=torch.float, device=pos_ei.device),
                neg_sampling_ratio=neg_ratio,
                batch_size=batch_size,
                shuffle=(split_name == 'train'),
                directed=True,
            )
            split_loaders[et] = loader
        loaders[split_name] = split_loaders

    return loaders