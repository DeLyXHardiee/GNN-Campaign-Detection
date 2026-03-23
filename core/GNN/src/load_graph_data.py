import sys
from pathlib import Path
from torch_geometric.datasets import IMDB
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected
from torch_geometric.data.storage import BaseStorage, NodeStorage, EdgeStorage

torch.serialization.add_safe_globals([HeteroData, BaseStorage, NodeStorage, EdgeStorage])

def load_imdb(root: str = "data/IMDB"):
    """
    Loads the PyG IMDB heterogeneous graph and returns the single HeteroData object.
    """
    root = str(Path(root))
    dataset = IMDB(root=root)
    return dataset[0]

def _default_hetero_pt_from_pipeline_config() -> str:
    """Resolve default .pt path from repo-root pipeline_config.json (graph + datasets)."""
    core_dir = Path(__file__).resolve().parent.parent.parent
    repo_root = core_dir.parent
    if str(core_dir) not in sys.path:
        sys.path.insert(0, str(core_dir))
    from config.pipeline_config import default_hetero_graph_pt_path

    return default_hetero_graph_pt_path(project_root=repo_root)


def load_hetero_pt(path: str = "../../graph/output/incidents-20260211-misp_hetero.pt", to_undirected=True):
    """
    Load a saved HeteroData object from a .pt file.
    When path is None, uses graph.output_dir and MISP basename from pipeline_config.json.
    """
    path = path or _default_hetero_pt_from_pipeline_config()
    path = str(Path(path).expanduser())
    data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(data, HeteroData):
        raise TypeError(f"Expected HeteroData in {path}, got {type(data)}")
    # Remove non-tensor node attributes so PyG loaders (e.g. LinkNeighborLoader) do not fail.
    # external_id is a list; get it from the companion .meta.json (email_attrs.external_id) when needed.
    if "email" in data.node_stores and hasattr(data["email"], "external_id"):
        del data["email"].external_id
    if to_undirected:
        return ToUndirected()(data)
    return data