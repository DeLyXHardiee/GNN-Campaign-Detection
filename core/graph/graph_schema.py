"""
Shared graph schema configuration used by both PyTorch-Geometric and Memgraph builders.

This module is the single source of truth for:
- Canonical node types and their labels in each backend
- Canonical relationship types and their labels in each backend
- Minimal node property conventions used by the Memgraph builder
- Lightweight feature strategies used by the PyTorch builder

Extend or modify this schema to evolve the graph without duplicating changes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class NodeMapping:
    """Mapping for a canonical node type to backend-specific labels and conventions.

    canonical: internal canonical name (stable across backends)
    pyg: node type label used in HeteroData for PyG
    memgraph: node label used in Memgraph
    memgraph_id_key: property used as a stable key when merging nodes in Memgraph
    feature_strategy: hint for PyG feature construction (kept minimal and optional)
    """

    canonical: str
    pyg: str
    memgraph: str
    memgraph_id_key: str
    feature_strategy: str


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


@dataclass(frozen=True)
class GraphSchema:
    nodes: Dict[str, NodeMapping]  # key by canonical name
    edges: Dict[str, EdgeMapping]  # key by canonical relationship name

    def node(self, canonical: str) -> NodeMapping:
        return self.nodes[canonical]

    def edge(self, canonical: str) -> EdgeMapping:
        return self.edges[canonical]

    def pyg_node_types(self) -> List[str]:
        return [n.pyg for n in self.nodes.values()]

    def pyg_edge_keys(self) -> List[Tuple[str, str, str]]:
        return [(self.nodes[e.src].pyg, e.rel_pyg, self.nodes[e.dst].pyg) for e in self.edges.values()]


# Default schema used by the project
DEFAULT_SCHEMA = GraphSchema(
    nodes={
        # Central hub per email
        "email": NodeMapping(
            canonical="email",
            pyg="email",
            memgraph="Email",
            memgraph_id_key="eid",
            feature_strategy="body_len",
        ),
        # People and addresses
        "sender": NodeMapping(
            canonical="sender",
            pyg="sender",
            memgraph="Sender",
            memgraph_id_key="key",
            feature_strategy="str_len",
        ),
        "receiver": NodeMapping(
            canonical="receiver",
            pyg="receiver",
            memgraph="Receiver",
            memgraph_id_key="key",
            feature_strategy="str_len",
        ),
        # Temporal bucket
        "week": NodeMapping(
            canonical="week",
            pyg="week",
            memgraph="Week",
            memgraph_id_key="key",
            feature_strategy="index",
        ),
        # Content/URLs
        "subject": NodeMapping(
            canonical="subject",
            pyg="subject",
            memgraph="Subject",
            memgraph_id_key="key",
            feature_strategy="str_len",
        ),
        "url": NodeMapping(
            canonical="url",
            pyg="url",
            memgraph="Url",
            memgraph_id_key="key",
            feature_strategy="str_len",
        ),
        "domain": NodeMapping(
            canonical="domain",
            pyg="domain",
            memgraph="Domain",
            memgraph_id_key="key",
            feature_strategy="str_len",
        ),
        "stem": NodeMapping(
            canonical="stem",
            pyg="stem",
            memgraph="Stem",
            memgraph_id_key="key",
            feature_strategy="str_len",
        ),
        # Domain extracted from email addresses
        "email_domain": NodeMapping(
            canonical="email_domain",
            pyg="email_domain",
            memgraph="EmailDomain",
            memgraph_id_key="key",
            feature_strategy="str_len",
        ),
    },
    edges={
        # Email -> components
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
        ),
        "in_week": EdgeMapping(
            canonical="in_week",
            src="email",
            rel_pyg="in_week",
            dst="week",
            memgraph_type="IN_WEEK",
            memgraph_left_label="Email",
            memgraph_left_key="eid",
            memgraph_right_label="Week",
            memgraph_right_key="key",
        ),
        "has_subject": EdgeMapping(
            canonical="has_subject",
            src="email",
            rel_pyg="has_subject",
            dst="subject",
            memgraph_type="HAS_SUBJECT",
            memgraph_left_label="Email",
            memgraph_left_key="eid",
            memgraph_right_label="Subject",
            memgraph_right_key="key",
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
        ),
        # URL -> components
        "url_has_domain": EdgeMapping(
            canonical="url_has_domain",
            src="url",
            rel_pyg="has_domain",
            dst="domain",
            memgraph_type="HAS_DOMAIN",
            memgraph_left_label="Url",
            memgraph_left_key="key",
            memgraph_right_label="Domain",
            memgraph_right_key="key",
        ),
        "url_has_stem": EdgeMapping(
            canonical="url_has_stem",
            src="url",
            rel_pyg="has_stem",
            dst="stem",
            memgraph_type="HAS_STEM",
            memgraph_left_label="Url",
            memgraph_left_key="key",
            memgraph_right_label="Stem",
            memgraph_right_key="key",
        ),
        # Email address entities -> their domain
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
        ),
    },
)


__all__ = [
    "NodeMapping",
    "EdgeMapping",
    "GraphSchema",
    "DEFAULT_SCHEMA",
]
