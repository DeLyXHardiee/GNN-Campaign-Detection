import torch

_torch_sparse_err = None
_pyg_lib_err = None
except ImportError as e:
    _torch_sparse_err = e
try:
    import pyg_lib  # noqa: F401
except ImportError as e:
    _pyg_lib_err = e

if _torch_sparse_err is not None and _pyg_lib_err is not None:
    import sys
    msg = (
        "NeighborSampler requires either 'pyg-lib' or 'torch-sparse'. Neither could be imported.\n"
        "  - torch_sparse: %s\n"
        "  - pyg_lib: %s\n"
        "Install in this environment using the wheel index that matches your PyTorch/CUDA.\n"
        "  Python: %s\n"
        "  Command: \"%s\" -m pip install torch-sparse -f https://data.pyg.org/whl/torch-<TORCH>+<CUDA>.html\n"
        "See core/GNN/docs/install_torch_sparse.md for TORCH/CUDA values."
    ) % (
        _torch_sparse_err,
        _pyg_lib_err,
        getattr(sys, "executable", "unknown"),
        getattr(sys, "executable", "unknown"),
    )
    raise ImportError(msg) from _torch_sparse_err
from torch_geometric.loader import LinkNeighborLoader, NeighborLoader

def make_link_loaders(train_graph, full_graph, train_pos, val_pos, test_pos,
                      edge_types, neg_ratio=1.0, batch_size=2048, fanout=[15, 10]):
    loaders = {}

    num_neighbors = {et: fanout for et in full_graph.edge_types}

    for split_name, pos_dict in [('train', train_pos), ('val', val_pos), ('test', test_pos)]:
        split_loaders = {}

        for et in edge_types:
            pos_ei = pos_dict[et]
            loader = LinkNeighborLoader(
                data=train_graph,
                num_neighbors=num_neighbors,
                edge_label_index=(et, pos_ei),
                edge_label=torch.ones(pos_ei.size(1), dtype=torch.float),
                neg_sampling_ratio=neg_ratio,
                batch_size=batch_size,
                shuffle=(split_name == 'train'),
                directed=True,
            )
            split_loaders[et] = loader
        loaders[split_name] = split_loaders

    return loaders