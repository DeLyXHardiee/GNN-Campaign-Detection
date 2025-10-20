"""
Visualization generator.

Creates visual representations of results.
"""

from typing import Optional, Any
import numpy as np
from pathlib import Path


class VisualizationGenerator:
    """
    Generates visualizations for pipeline results.
    """

    def __init__(self, output_dir: str = "visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def visualize_graph(self, graph: Any) -> Optional[str]:
        """
        Visualize the graph structure.
        Baseline implementation - in production use networkx/matplotlib.

        Args:
            graph: Graph object

        Returns:
            Path to saved visualization
        """
        try:
            import networkx as nx
            import matplotlib.pyplot as plt

            if not isinstance(graph, nx.Graph):
                return None

            output_path = self.output_dir / "graph.png"

            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(graph)
            nx.draw(
                graph,
                pos,
                node_size=50,
                node_color='lightblue',
                edge_color='gray',
                alpha=0.6,
                with_labels=False
            )
            plt.title("Graph Structure")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()

            return str(output_path)

        except Exception as e:
            print(f"Failed to generate graph visualization: {e}")
            return None

    def visualize_embeddings(
        self,
        embeddings: np.ndarray,
        clusters: np.ndarray
    ) -> Optional[str]:
        """
        Visualize embeddings in 2D using dimensionality reduction.
        Baseline implementation - in production use UMAP/t-SNE.

        Args:
            embeddings: Node embeddings
            clusters: Cluster labels

        Returns:
            Path to saved visualization
        """
        try:
            import matplotlib.pyplot as plt

            output_path = self.output_dir / "embeddings.png"

            # Baseline: use PCA for dimensionality reduction
            # In production: from sklearn.decomposition import PCA
            # pca = PCA(n_components=2)
            # embeddings_2d = pca.fit_transform(embeddings)

            # For now, just use first two dimensions if available
            if embeddings.shape[1] >= 2:
                embeddings_2d = embeddings[:, :2]
            else:
                return None

            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(
                embeddings_2d[:, 0],
                embeddings_2d[:, 1],
                c=clusters,
                cmap='viridis',
                alpha=0.6
            )
            plt.colorbar(scatter, label='Cluster')
            plt.title("Embedding Visualization (2D Projection)")
            plt.xlabel("Dimension 1")
            plt.ylabel("Dimension 2")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()

            return str(output_path)

        except Exception as e:
            print(f"Failed to generate embedding visualization: {e}")
            return None

    def visualize_cluster_distribution(
        self,
        clusters: np.ndarray
    ) -> Optional[str]:
        """
        Visualize cluster size distribution.

        Args:
            clusters: Cluster labels

        Returns:
            Path to saved visualization
        """
        try:
            import matplotlib.pyplot as plt

            output_path = self.output_dir / "cluster_distribution.png"

            unique, counts = np.unique(clusters, return_counts=True)

            plt.figure(figsize=(10, 6))
            plt.bar(unique, counts, color='steelblue', alpha=0.7)
            plt.xlabel("Cluster ID")
            plt.ylabel("Number of Nodes")
            plt.title("Cluster Size Distribution")
            plt.grid(axis='y', alpha=0.3)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()

            return str(output_path)

        except Exception as e:
            print(f"Failed to generate cluster distribution: {e}")
            return None

    def visualize_metrics(
        self,
        metrics: dict,
        output_name: str = "metrics"
    ) -> Optional[str]:
        """
        Visualize metrics as a table or chart.

        Args:
            metrics: Dictionary of metrics
            output_name: Name for output file

        Returns:
            Path to saved visualization
        """
        try:
            import matplotlib.pyplot as plt

            output_path = self.output_dir / f"{output_name}.png"

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.axis('tight')
            ax.axis('off')

            # Convert metrics to table format
            table_data = [[k, str(v)] for k, v in metrics.items()]

            table = ax.table(
                cellText=table_data,
                colLabels=['Metric', 'Value'],
                cellLoc='left',
                loc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)

            plt.title("Pipeline Metrics", pad=20)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()

            return str(output_path)

        except Exception as e:
            print(f"Failed to generate metrics visualization: {e}")
            return None
