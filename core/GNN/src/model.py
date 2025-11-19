import torch
from torch import nn
from torch_geometric.nn import SAGEConv, to_hetero
# -----------------------------
# Hetero GraphSAGE via to_hetero
# -----------------------------
class SAGEBackbone(nn.Module):
    def __init__(self, hidden=128, out=128, layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        # first layer (in channels unknown -> -1)
        self.layers.append(SAGEConv((-1, -1), hidden))
        # middle hidden layers
        for _ in range(layers - 2):
            self.layers.append(SAGEConv((hidden, hidden), hidden))
        # final
        if layers > 1:
            self.layers.append(SAGEConv((hidden, hidden), out))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        h = x
        for i, conv in enumerate(self.layers):
            h = conv(h, edge_index)
            if i < len(self.layers) - 1:
                h = torch.relu(h)
                h = self.dropout(h)
        return h

class HeteroSAGE(nn.Module):
    def __init__(self, metadata, hidden=128, out=128, layers=2, dropout=0.1):
        super().__init__()
        self.gnn = to_hetero(SAGEBackbone(hidden, out, layers, dropout), metadata)

    def forward(self, x_dict, edge_index_dict):
        return self.gnn(x_dict, edge_index_dict)  # returns dict: {node_type: embeddings}
    

class DotPredictor(nn.Module):
    def forward(self, src, dst):
        return (src * dst).sum(dim=-1)  # logits

class MLPredictor(nn.Module):
    def __init__(self, d, h=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2*d, h),
            nn.ReLU(),
            nn.Linear(h, 1),
        )
    def forward(self, src, dst):
        return self.net(torch.cat([src, dst], dim=-1)).squeeze(-1)
    

