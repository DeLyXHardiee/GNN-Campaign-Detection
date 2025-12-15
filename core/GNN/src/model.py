import torch
from torch import nn
from torch_geometric.nn import SAGEConv, to_hetero
import torch.nn.functional as F

class SAGEBackbone(nn.Module):
    def __init__(self, hidden=128, out=128, layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv((-1, -1), hidden))
        for _ in range(layers - 2):
            self.layers.append(SAGEConv((hidden, hidden), hidden))
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
        h = F.normalize(h, p=2, dim=-1)
        return h


class HeteroSAGE(nn.Module):
    def __init__(self, metadata, hidden=128, out=128, layers=2, dropout=0.1):
        super().__init__()
        self.gnn = to_hetero(SAGEBackbone(hidden, out, layers, dropout), metadata)

    def forward(self, x_dict, edge_index_dict):
        return self.gnn(x_dict, edge_index_dict)
    

class DotPredictor(nn.Module):
    def forward(self, src, dst, edge_types=None):
        return (src * dst).sum(dim=-1)

class MLPredictor(nn.Module):
    def __init__(self, d, h=128, edge_types=None, dropout=0.3, use_dropout=True):
        super().__init__()
        layers = [
            nn.Linear(2 * d, h),
            nn.ReLU(),
        ]
        if use_dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(h, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, src, dst):
        return self.net(torch.cat([src, dst], dim=-1)).squeeze(-1)
    

import math
import torch
from torch import nn

class DistMultPredictor(nn.Module):
    def __init__(self, dim, edge_types):
        super().__init__()
        self.dim = dim
        self.rel_params = nn.ParameterDict()

        sigma = 1.0 / math.sqrt(dim)

        for et in edge_types:
            key = "__".join(et)
            param = nn.Parameter(torch.empty(dim))
            nn.init.normal_(param, mean=0.0, std=sigma)
            self.rel_params[key] = param

    def forward(self, src, dst, edge_type):
        key = "__".join(edge_type)
        r = self.rel_params[key]
        return (src * r * dst).sum(dim=-1)


    
