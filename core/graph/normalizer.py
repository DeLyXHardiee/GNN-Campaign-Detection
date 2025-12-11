import torch

def normalize_graph(data):
    """
    Standardize features: zero mean, unit variance per feature dim.
    Expects a PyG HeteroData object.
    """
    for ntype in data.node_types:
        if 'x' not in data[ntype]:
            continue

        x = data[ntype].x

        if x.dtype != torch.float32:
            x = x.float()

        mu = x.mean(dim=0)
        sigma = x.std(dim=0).clamp_min(1e-6)

        data[ntype].x = (x - mu) / sigma
    
    return data


