import torch
from torch_geometric.loader import LinkNeighborLoader, NeighborLoader
# -----------------------------
# Mini-batch loaders (one per supervised relation)
# -----------------------------

# A factory that builds mini batches for training called 'loaders' for each supervised relation in edge_types
def make_link_loaders(train_graph, full_graph, train_pos, val_pos, test_pos,
                      edge_types, neg_ratio=1.0, batch_size=2048, fanout=[15, 10]):
    loaders = {}

    # Builds a mapping relation → [15,10] so the sampler knows how many neighbors to pull per relation, per hop.
    # In this instance it's the same for all relations
    num_neighbors = {et: fanout for et in full_graph.edge_types}

    # We’ll build a set of loaders for each split; pos_dict is that split’s positive (real) edges per relation.
    for split_name, pos_dict in [('train', train_pos), ('val', val_pos), ('test', test_pos)]:
        split_loaders = {}

        #Pick the supervised relation et (e.g., ('movie','to','actor')), and grab the positive edge_index pos_ei for this split.
        #This is the set of edges we want the model to score as 1/true
        for et in edge_types:
            pos_ei = pos_dict[et]
            # Build a loader over the training graph for all splits (neighbors come from train_graph)
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