"""
Graph generation stage.

Generates graph representations from MISP objects.
Implements the Builder pattern for complex graph construction.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import networkx as nx

from src.pipeline.base import PipelineStageInterface, PipelineContext
from src.misp.converter import MISPObject, MISPRelationship


class GraphBuilder(ABC):
    """
    Abstract base class for graph builders (Builder pattern).
    """

    @abstractmethod
    def build_graph(self, misp_objects: List[MISPObject]) -> Any:
        """
        Build a graph from MISP objects.

        Args:
            misp_objects: List of MISP objects

        Returns:
            Graph object
        """
        pass


class NetworkXGraphBuilder(GraphBuilder):
    """
    Builds NetworkX graphs from MISP objects.
    """

    def __init__(
        self,
        directed: bool = True,
        include_node_attributes: bool = True,
        include_edge_attributes: bool = True
    ):
        self.directed = directed
        self.include_node_attributes = include_node_attributes
        self.include_edge_attributes = include_edge_attributes

    def build_graph(self, misp_objects: List[MISPObject]) -> nx.Graph:
        """
        Build a NetworkX graph.

        Args:
            misp_objects: List of MISP objects

        Returns:
            NetworkX graph
        """
        # Create graph
        G = nx.DiGraph() if self.directed else nx.Graph()

        # Add nodes
        for obj in misp_objects:
            node_attrs = {}

            if self.include_node_attributes:
                node_attrs = {
                    'type': obj.object_type,
                    **obj.attributes
                }

            G.add_node(obj.uuid, **node_attrs)

        # Add edges from relationships
        for obj in misp_objects:
            for rel in obj.relationships:
                edge_attrs = {}

                if self.include_edge_attributes:
                    edge_attrs = {
                        'relationship_type': rel.relationship_type
                    }

                G.add_edge(
                    rel.source_uuid,
                    rel.target_uuid,
                    **edge_attrs
                )

        return G


class PyTorchGeometricGraphBuilder(GraphBuilder):
    """
    Builds PyTorch Geometric Data objects from MISP objects.
    This is a placeholder - actual implementation would require torch_geometric.
    """

    def __init__(
        self,
        feature_columns: Optional[List[str]] = None,
        label_column: Optional[str] = None
    ):
        self.feature_columns = feature_columns
        self.label_column = label_column

    def build_graph(self, misp_objects: List[MISPObject]) -> Any:
        """
        Build a PyTorch Geometric Data object.

        This is a baseline implementation.
        In production, this would create torch_geometric.data.Data objects.

        Args:
            misp_objects: List of MISP objects

        Returns:
            Graph data structure (dict placeholder)
        """
        # Create mapping from UUID to index
        uuid_to_idx = {obj.uuid: idx for idx, obj in enumerate(misp_objects)}

        # Extract features
        node_features = []
        for obj in misp_objects:
            features = []
            if self.feature_columns:
                for col in self.feature_columns:
                    features.append(obj.attributes.get(col, 0))
            node_features.append(features)

        # Extract edges
        edge_list = []
        for obj in misp_objects:
            for rel in obj.relationships:
                source_idx = uuid_to_idx.get(rel.source_uuid)
                target_idx = uuid_to_idx.get(rel.target_uuid)

                if source_idx is not None and target_idx is not None:
                    edge_list.append([source_idx, target_idx])

        # Placeholder structure
        # In production: torch_geometric.data.Data(x=features, edge_index=edges)
        return {
            'node_features': node_features,
            'edge_index': edge_list,
            'num_nodes': len(misp_objects)
        }


class GraphBuilderFactory:
    """
    Factory for creating graph builders.
    Implements the Factory pattern.
    """

    @staticmethod
    def create(builder_type: str, **kwargs) -> GraphBuilder:
        """
        Create a graph builder instance.

        Args:
            builder_type: Type of builder ('networkx', 'pytorch_geometric')
            **kwargs: Additional arguments for the builder

        Returns:
            GraphBuilder instance
        """
        if builder_type == 'networkx':
            return NetworkXGraphBuilder(**kwargs)
        elif builder_type == 'pytorch_geometric':
            return PyTorchGeometricGraphBuilder(**kwargs)
        else:
            raise ValueError(f"Unknown builder type: {builder_type}")


class GraphGeneratorStage(PipelineStageInterface):
    """
    Pipeline stage for generating graphs from MISP objects.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        builder_type = config.get('builder_type', 'networkx')
        builder_kwargs = config.get('builder_kwargs', {})

        self.builder = GraphBuilderFactory.create(builder_type, **builder_kwargs)

    def validate(self, context: PipelineContext) -> bool:
        """Validate that MISP objects exist."""
        return (
            context.misp_objects is not None
            and len(context.misp_objects) > 0
        )

    def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Generate graph and update context.

        Args:
            context: Current pipeline context

        Returns:
            Updated context with graph
        """
        self.notify_observers("graph_generation_started", {})

        try:
            # Build graph
            graph = self.builder.build_graph(context.misp_objects)

            # Collect metadata
            if isinstance(graph, nx.Graph):
                num_nodes = graph.number_of_nodes()
                num_edges = graph.number_of_edges()
            elif isinstance(graph, dict):
                num_nodes = graph.get('num_nodes', 0)
                num_edges = len(graph.get('edge_index', []))
            else:
                num_nodes = 0
                num_edges = 0

            self.notify_observers("graph_generation_completed", {
                "nodes": num_nodes,
                "edges": num_edges
            })

            context.graph = graph
            context.metadata['graph_nodes'] = num_nodes
            context.metadata['graph_edges'] = num_edges

        except Exception as e:
            self.notify_observers("graph_generation_failed", {
                "error": str(e)
            })
            raise

        return context
