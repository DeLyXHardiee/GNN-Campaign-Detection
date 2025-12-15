import torch
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