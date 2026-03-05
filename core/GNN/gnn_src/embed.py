import torch
from torch_geometric.loader import NeighborLoader

@torch.no_grad()
def get_primary_embeddings(DEVICE, model, data, primary_ntype='movie', batch_size=4096):
    model.eval()
    num_neighbors = {et: [-1, -1] for et in data.edge_types}
    loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        input_nodes=(primary_ntype, torch.arange(data[primary_ntype].num_nodes)),
        batch_size=batch_size, shuffle=False
    )
    out = None
    for batch in loader:
        batch = batch.to(DEVICE)
        h_dict = model(batch.x_dict, batch.edge_index_dict)
        idx = batch[primary_ntype].n_id
        if out is None:
            D = h_dict[primary_ntype].size(-1)
            out = torch.empty((data[primary_ntype].num_nodes, D))
        out[idx] = h_dict[primary_ntype].cpu()
    return out

@torch.no_grad()
def embed_with_graph(DEVICE, model, graph):
    model.eval()
    if DEVICE is not None:
        graph = graph.to(DEVICE)
    return model(graph.x_dict, graph.edge_index_dict)

def normalize_dict(h_dict):
    return {k: torch.nn.functional.normalize(v, p=2, dim=1) for k, v in h_dict.items()}

@torch.no_grad()
def export_embeddings(DEVICE, model, train_graph, primary_ntype='movie', layers=2, batch_size=4096):
    model.eval()
    num_neighbors = {et: [-1] * layers for et in train_graph.edge_types}
    loader = NeighborLoader(
        train_graph,
        num_neighbors=num_neighbors,
        input_nodes=(primary_ntype, torch.arange(train_graph[primary_ntype].num_nodes)),
        batch_size=batch_size, shuffle=False
    )
    out = None
    for batch in loader:
        batch = batch.to(DEVICE)
        h_dict = model(batch.x_dict, batch.edge_index_dict)
        idx = batch[primary_ntype].n_id
        if out is None:
            D = h_dict[primary_ntype].size(-1)
            out = torch.empty((train_graph[primary_ntype].num_nodes, D), device=DEVICE)
        out[idx] = h_dict[primary_ntype].detach().cpu()
    return out

