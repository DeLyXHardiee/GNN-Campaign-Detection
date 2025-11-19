from pathlib import Path
from torch_geometric.datasets import IMDB
import torch
from torch_geometric.data import HeteroData

def load_imdb(root: str = "data/IMDB"):
    """
    Loads the PyG IMDB heterogeneous graph and returns the single HeteroData object.
    """
    root = str(Path(root))  # normalize
    dataset = IMDB(root=root)
    return dataset[0]

def load_hetero_pt(path: str = "data/email/trec07_misp_hetero.pt"):
    """
    Load a saved HeteroData object from a .pt file.
    """
    path = str(Path(path).expanduser())
    data = torch.load(path, map_location="cpu")   # or your DEVICE
    if not isinstance(data, HeteroData):
        raise TypeError(f"Expected HeteroData in {path}, got {type(data)}")
    return data


