"""
Utilities to keep :class:`torch_geometric.data.HeteroData` consistent for GNN training.

Excluded node types (see ``filter_graph_ir``) may leave behind empty node stores with no
``x`` and no incident edges. Those stores still appear in :meth:`HeteroData.metadata`,
which breaks :func:`torch_geometric.nn.to_hetero` with ``SAGEConv``: PyG expects every
node type in the metadata to receive updates during message passing.
"""

from __future__ import annotations

from torch_geometric.data import HeteroData


def prune_heterodata_for_message_passing(data: HeteroData) -> HeteroData:
    """
    Drop empty edge relations and node stores that do not participate in any non-empty
    edge (or that have no feature tensor). Safe to call after ``ToUndirected`` and
    :func:`normalize_graph`.
    """
    for et in list(data.edge_types):
        store = data[et]
        ei = store.get("edge_index", None)
        if ei is None or ei.numel() == 0:
            del data[et]

    incident: set[str] = set()
    for et in data.edge_types:
        incident.add(et[0])
        incident.add(et[2])

    for nt in list(data.node_types):
        if nt not in incident:
            del data[nt]
            continue
        st = data[nt]
        if "x" not in st or st["x"] is None:
            del data[nt]

    for et in list(data.edge_types):
        src, _, dst = et
        if src not in data.node_types or dst not in data.node_types:
            del data[et]

    return data
