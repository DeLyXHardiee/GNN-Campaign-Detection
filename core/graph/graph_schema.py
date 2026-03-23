"""
Shared graph schema configuration used by both PyTorch-Geometric and Memgraph builders.

This module is the single source of truth for:
- Canonical node types and their labels in each backend
- Canonical relationship types and their labels in each backend
- Minimal node property conventions used by the Memgraph builder
- Lightweight feature strategies used by the PyTorch builder
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class NodeMapping:
    """Mapping for a canonical node type to backend-specific labels and conventions.

    canonical: internal canonical name
    pyg: node type label used in HeteroData for PyG
    memgraph: node label used in Memgraph
    memgraph_id_key: property used as a stable key when merging nodes in Memgraph
    feature_strategy: hint for feature construction
    extra_attr_keys: optional attributes to merge into base node features
    """

    canonical: str
    pyg: str
    memgraph: str
    memgraph_id_key: str
    feature_strategy: str
    extra_attr_keys: Tuple[str, ...] = ()


@dataclass(frozen=True)
class EdgeMapping:
    """Mapping for a canonical relationship type to backend-specific labels and conventions.

    canonical: internal canonical name
    src: canonical name of source node type
    rel_pyg: relation name used in HeteroData edge key
    dst: canonical name of destination node type
    memgraph_type: relationship type used in Memgraph
    memgraph_left_label: label of left node in Memgraph
    memgraph_left_key: property of left node used to match
    memgraph_right_label: label of right node in Memgraph
    memgraph_right_key: property of right node used to match
    edge_strategy: optional hint describing how this edge is materialized.
    """

    canonical: str
    src: str
    rel_pyg: str
    dst: str
    memgraph_type: str
    memgraph_left_label: str
    memgraph_left_key: str
    memgraph_right_label: str
    memgraph_right_key: str
    edge_strategy: str = "default"


@dataclass(frozen=True)
class GraphSchema:
    nodes: Dict[str, NodeMapping]  # key by canonical name
    edges: Dict[str, EdgeMapping]  # key by canonical relationship name
    collapse_rules: Tuple[Tuple[str, str, str], ...] = field(default_factory=tuple)

    def node(self, canonical: str) -> NodeMapping:
        return self.nodes[canonical]

    def edge(self, canonical: str) -> EdgeMapping:
        return self.edges[canonical]

    def pyg_node_types(self) -> List[str]:
        return [n.pyg for n in self.nodes.values()]

    def pyg_edge_keys(self) -> List[Tuple[str, str, str]]:
        return [(self.nodes[e.src].pyg, e.rel_pyg, self.nodes[e.dst].pyg) for e in self.edges.values()]


def validate_schema(schema: GraphSchema) -> None:
    """Validate graph schema consistency and raise ValueError when invalid."""
    node_keys = list(schema.nodes.keys())
    edge_keys = list(schema.edges.keys())
    if len(node_keys) != len(set(node_keys)):
        raise ValueError("Schema contains duplicate node canonical keys.")
    if len(edge_keys) != len(set(edge_keys)):
        raise ValueError("Schema contains duplicate edge canonical keys.")

    pyg_nodes = [n.pyg for n in schema.nodes.values()]
    mem_nodes = [n.memgraph for n in schema.nodes.values()]
    if len(pyg_nodes) != len(set(pyg_nodes)):
        raise ValueError("Schema contains duplicate PyG node labels.")
    if len(mem_nodes) != len(set(mem_nodes)):
        raise ValueError("Schema contains duplicate Memgraph node labels.")

    for edge_key, edge in schema.edges.items():
        if edge.src not in schema.nodes:
            raise ValueError(f"Edge '{edge_key}' references unknown src node '{edge.src}'.")
        if edge.dst not in schema.nodes:
            raise ValueError(f"Edge '{edge_key}' references unknown dst node '{edge.dst}'.")
        left = schema.nodes[edge.src]
        right = schema.nodes[edge.dst]
        if left.memgraph != edge.memgraph_left_label:
            raise ValueError(
                f"Edge '{edge_key}' left label '{edge.memgraph_left_label}' does not match "
                f"node '{edge.src}' label '{left.memgraph}'."
            )
        if right.memgraph != edge.memgraph_right_label:
            raise ValueError(
                f"Edge '{edge_key}' right label '{edge.memgraph_right_label}' does not match "
                f"node '{edge.dst}' label '{right.memgraph}'."
            )
        if left.memgraph_id_key != edge.memgraph_left_key:
            raise ValueError(
                f"Edge '{edge_key}' left key '{edge.memgraph_left_key}' does not match "
                f"node '{edge.src}' key '{left.memgraph_id_key}'."
            )
        if right.memgraph_id_key != edge.memgraph_right_key:
            raise ValueError(
                f"Edge '{edge_key}' right key '{edge.memgraph_right_key}' does not match "
                f"node '{edge.dst}' key '{right.memgraph_id_key}'."
            )

    for parent, child, edge_key in schema.collapse_rules:
        if parent not in schema.nodes:
            raise ValueError(f"Collapse rule references unknown parent node '{parent}'.")
        if child not in schema.nodes:
            raise ValueError(f"Collapse rule references unknown child node '{child}'.")
        if edge_key not in schema.edges:
            raise ValueError(f"Collapse rule references unknown edge '{edge_key}'.")
        edge = schema.edges[edge_key]
        if edge.src != parent or edge.dst != child:
            raise ValueError(
                f"Collapse rule ({parent}, {child}, {edge_key}) mismatches edge endpoints "
                f"({edge.src}, {edge.dst})."
            )


DEFAULT_SCHEMA = GraphSchema(
    nodes={
        "email": NodeMapping(
            canonical="email",
            pyg="email",
            memgraph="Email",
            memgraph_id_key="eid",
            feature_strategy="body_len",
        ),
        "sender": NodeMapping(
            canonical="sender",
            pyg="sender",
            memgraph="Sender",
            memgraph_id_key="key",
            feature_strategy="str_len",
            extra_attr_keys=("docfreq",),
        ),
        "receiver": NodeMapping(
            canonical="receiver",
            pyg="receiver",
            memgraph="Receiver",
            memgraph_id_key="key",
            feature_strategy="str_len",
            extra_attr_keys=("docfreq",),
        ),
        "url": NodeMapping(
            canonical="url",
            pyg="url",
            memgraph="Url",
            memgraph_id_key="key",
            feature_strategy="str_len",
            extra_attr_keys=("x_lex", "docfreq"),
        ),
        "domain": NodeMapping(
            canonical="domain",
            pyg="domain",
            memgraph="Domain",
            memgraph_id_key="key",
            feature_strategy="str_len",
            extra_attr_keys=("x_lex", "docfreq"),
        ),
        "stem": NodeMapping(
            canonical="stem",
            pyg="stem",
            memgraph="Stem",
            memgraph_id_key="key",
            feature_strategy="str_len",
            extra_attr_keys=("x_lex", "docfreq"),
        ),
        "email_domain": NodeMapping(
            canonical="email_domain",
            pyg="email_domain",
            memgraph="EmailDomain",
            memgraph_id_key="key",
            feature_strategy="str_len",
            extra_attr_keys=("x_lex", "docfreq_sender", "docfreq_receiver"),
        ),
        "attachment": NodeMapping(
            canonical="attachment",
            pyg="attachment",
            memgraph="Attachment",
            memgraph_id_key="key",
            feature_strategy="str_len",
            extra_attr_keys=("docfreq",),
        ),
        "origin_ip": NodeMapping(
            canonical="origin_ip",
            pyg="origin_ip",
            memgraph="OriginIp",
            memgraph_id_key="key",
            feature_strategy="str_len",
            extra_attr_keys=("docfreq",),
        ),
        "received_host": NodeMapping(
            canonical="received_host",
            pyg="received_host",
            memgraph="ReceivedHost",
            memgraph_id_key="key",
            feature_strategy="str_len",
            extra_attr_keys=("docfreq",),
        ),
        "return_path_email": NodeMapping(
            canonical="return_path_email",
            pyg="return_path_email",
            memgraph="ReturnPathEmail",
            memgraph_id_key="key",
            feature_strategy="str_len",
            extra_attr_keys=("docfreq",),
        ),
        "return_path_domain": NodeMapping(
            canonical="return_path_domain",
            pyg="return_path_domain",
            memgraph="ReturnPathDomain",
            memgraph_id_key="key",
            feature_strategy="str_len",
            extra_attr_keys=("docfreq",),
        ),
    },
    edges={
        "has_sender": EdgeMapping(
            canonical="has_sender",
            src="email",
            rel_pyg="has_sender",
            dst="sender",
            memgraph_type="HAS_SENDER",
            memgraph_left_label="Email",
            memgraph_left_key="eid",
            memgraph_right_label="Sender",
            memgraph_right_key="key",
            edge_strategy="email_to_entity",
        ),
        "has_receiver": EdgeMapping(
            canonical="has_receiver",
            src="email",
            rel_pyg="has_receiver",
            dst="receiver",
            memgraph_type="HAS_RECEIVER",
            memgraph_left_label="Email",
            memgraph_left_key="eid",
            memgraph_right_label="Receiver",
            memgraph_right_key="key",
            edge_strategy="email_to_entity",
        ),
        "has_url": EdgeMapping(
            canonical="has_url",
            src="email",
            rel_pyg="has_url",
            dst="url",
            memgraph_type="HAS_URL",
            memgraph_left_label="Email",
            memgraph_left_key="eid",
            memgraph_right_label="Url",
            memgraph_right_key="key",
            edge_strategy="email_to_entity",
        ),
        "has_domain": EdgeMapping(
            canonical="has_domain",
            src="email",
            rel_pyg="has_domain",
            dst="domain",
            memgraph_type="HAS_DOMAIN",
            memgraph_left_label="Email",
            memgraph_left_key="eid",
            memgraph_right_label="Domain",
            memgraph_right_key="key",
            edge_strategy="email_to_domain_from_urls",
        ),
        "has_stem": EdgeMapping(
            canonical="has_stem",
            src="email",
            rel_pyg="has_stem",
            dst="stem",
            memgraph_type="HAS_STEM",
            memgraph_left_label="Email",
            memgraph_left_key="eid",
            memgraph_right_label="Stem",
            memgraph_right_key="key",
            edge_strategy="email_to_stem_from_urls",
        ),
        "sender_from_domain": EdgeMapping(
            canonical="sender_from_domain",
            src="sender",
            rel_pyg="from_domain",
            dst="email_domain",
            memgraph_type="FROM_DOMAIN",
            memgraph_left_label="Sender",
            memgraph_left_key="key",
            memgraph_right_label="EmailDomain",
            memgraph_right_key="key",
            edge_strategy="entity_to_entity",
        ),
        "receiver_from_domain": EdgeMapping(
            canonical="receiver_from_domain",
            src="receiver",
            rel_pyg="from_domain",
            dst="email_domain",
            memgraph_type="FROM_DOMAIN",
            memgraph_left_label="Receiver",
            memgraph_left_key="key",
            memgraph_right_label="EmailDomain",
            memgraph_right_key="key",
            edge_strategy="entity_to_entity",
        ),
        "has_attachment": EdgeMapping(
            canonical="has_attachment",
            src="email",
            rel_pyg="has_attachment",
            dst="attachment",
            memgraph_type="HAS_ATTACHMENT",
            memgraph_left_label="Email",
            memgraph_left_key="eid",
            memgraph_right_label="Attachment",
            memgraph_right_key="key",
            edge_strategy="email_to_entity",
        ),
        "has_origin_ip": EdgeMapping(
            canonical="has_origin_ip",
            src="email",
            rel_pyg="has_origin_ip",
            dst="origin_ip",
            memgraph_type="HAS_ORIGIN_IP",
            memgraph_left_label="Email",
            memgraph_left_key="eid",
            memgraph_right_label="OriginIp",
            memgraph_right_key="key",
            edge_strategy="email_to_entity",
        ),
        "has_received_host": EdgeMapping(
            canonical="has_received_host",
            src="email",
            rel_pyg="has_received_host",
            dst="received_host",
            memgraph_type="HAS_RECEIVED_HOST",
            memgraph_left_label="Email",
            memgraph_left_key="eid",
            memgraph_right_label="ReceivedHost",
            memgraph_right_key="key",
            edge_strategy="email_to_entity",
        ),
        "has_return_path_email": EdgeMapping(
            canonical="has_return_path_email",
            src="email",
            rel_pyg="has_return_path_email",
            dst="return_path_email",
            memgraph_type="HAS_RETURN_PATH_EMAIL",
            memgraph_left_label="Email",
            memgraph_left_key="eid",
            memgraph_right_label="ReturnPathEmail",
            memgraph_right_key="key",
            edge_strategy="email_to_entity",
        ),
        "has_return_path_domain": EdgeMapping(
            canonical="has_return_path_domain",
            src="email",
            rel_pyg="has_return_path_domain",
            dst="return_path_domain",
            memgraph_type="HAS_RETURN_PATH_DOMAIN",
            memgraph_left_label="Email",
            memgraph_left_key="eid",
            memgraph_right_label="ReturnPathDomain",
            memgraph_right_key="key",
            edge_strategy="email_to_entity",
        ),
    },
    collapse_rules=(
        ("sender", "email_domain", "sender_from_domain"),
        ("receiver", "email_domain", "receiver_from_domain"),
        ("email", "sender", "has_sender"),
        ("email", "receiver", "has_receiver"),
        ("email", "url", "has_url"),
        ("email", "domain", "has_domain"),
        ("email", "stem", "has_stem"),
    ),
)

validate_schema(DEFAULT_SCHEMA)


__all__ = [
    "NodeMapping",
    "EdgeMapping",
    "GraphSchema",
    "validate_schema",
    "DEFAULT_SCHEMA",
]
