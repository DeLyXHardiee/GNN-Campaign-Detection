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

        # 1) ensure float32
        if x.dtype != torch.float32:
            x = x.float()

        # 2) standardize features: zero mean, unit variance per feature dim
        mu = x.mean(dim=0)
        sigma = x.std(dim=0).clamp_min(1e-6)  # avoid division by zero

        data[ntype].x = (x - mu) / sigma
    
    return data


