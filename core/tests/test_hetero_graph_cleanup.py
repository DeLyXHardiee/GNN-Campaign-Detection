import torch
from torch_geometric.data import HeteroData

from core.graph.hetero_graph_cleanup import drop_inactive_hetero_node_types


def test_drop_inactive_removes_empty_placeholder_node_types():
    data = HeteroData()
    data["email"].x = torch.randn(3, 4)
    data["sender"].x = torch.randn(2, 2)
    data["email", "has_sender", "sender"].edge_index = torch.tensor([[0, 1, 2], [0, 1, 0]], dtype=torch.long)
    _ = data["origin_ip"]

    cleaned = drop_inactive_hetero_node_types(data)
    assert "origin_ip" not in cleaned.node_types
    assert "email" in cleaned.node_types
    assert "sender" in cleaned.node_types
