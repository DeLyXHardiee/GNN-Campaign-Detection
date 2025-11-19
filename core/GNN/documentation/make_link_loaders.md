This function is a little complicated, but I will try to explain to summarize it in easy to understand terms.

**Arguments for make_link_loaders**:
- **train_graph**: the part of the graph that the model will be trained on. Accounts for about 80% of the edges in the graph.
- **full_graph**: used to get all edge_types in the graph.
- **train_pos**: row index of all the edges used for training
- **val_pos**: row index of all the edges used for validation
- **test_pos**: row index of all the edges used for validation
- **edge_types**: all edge types
- **neg_ratio**: the amount of false edges per. real edge used for training.
- **batch size**: the number of real edges in a batch.
- **fan-out**: the number of neighbors sampled in each hop (first 15, then 10)

The function builds a loader for each edge_type in the graph.
We build seperate loaders for each split: the training set, the validation set and the test set.

Then for each edge type (et) [('movie', 'to', 'director'), ('movie', 'to', 'actor'), ('director', 'to', 'movie'), ('actor', 'to', 'movie')], we get all the real edge row indexes for that type (pos_ei = pos_dict[et]).

Then we build the link-loader, called **NeighboorLinkLoader**. It contains:

- **data**: the training graph
- **num_neighbors**: the fan-out or the number of neighboors sampled in each hop for that specific edge_type.
- **edge_label_index**: the edge type along with all it's row indexes.
- **neg_ratio**: the amount of false edges per. real edge used for training.
- **batch size**: the number of edges in each batch
- **shuffle**: if it should shuffle the edges, we only do this for training for some reason.
- **directed**: a boolean specyfing if the graph is directed or not.

The we save the loader for that edge_type (split_loaders[et] = loader).
Then we save all the loaders for all the edge types in that split (loaders[split_name] = split_loaders).
