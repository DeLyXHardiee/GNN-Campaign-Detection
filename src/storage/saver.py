"""
Result storage stage.

Stores results including metrics and visualizations.
Implements the Repository pattern for data persistence.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List
import json
from pathlib import Path
from datetime import datetime

from src.pipeline.base import PipelineStageInterface, PipelineContext


class ResultRepository(ABC):
    """
    Abstract base class for result storage (Repository pattern).
    """

    @abstractmethod
    def save_metrics(self, metrics: Dict[str, Any], path: str) -> None:
        """Save numerical metrics."""
        pass

    @abstractmethod
    def save_visualizations(self, visualizations: Dict[str, Any], path: str) -> None:
        """Save visualization files."""
        pass

    @abstractmethod
    def save_embeddings(self, embeddings: Any, path: str) -> None:
        """Save embeddings."""
        pass

    @abstractmethod
    def save_clusters(self, clusters: Any, path: str) -> None:
        """Save cluster assignments."""
        pass


class FileSystemRepository(ResultRepository):
    """
    File system-based result storage.
    """

    def __init__(self, base_path: str = "results"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save_metrics(self, metrics: Dict[str, Any], path: str) -> None:
        """Save metrics as JSON."""
        full_path = self.base_path / path
        full_path.parent.mkdir(parents=True, exist_ok=True)

        with open(full_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)

    def save_visualizations(self, visualizations: Dict[str, Any], path: str) -> None:
        """
        Save visualizations.
        Visualizations should be pre-rendered as files.
        """
        # This method receives paths to already-created visualization files
        # and could copy them to the results directory if needed
        pass

    def save_embeddings(self, embeddings: Any, path: str) -> None:
        """Save embeddings as numpy array."""
        import numpy as np

        full_path = self.base_path / path
        full_path.parent.mkdir(parents=True, exist_ok=True)

        np.save(full_path, embeddings)

    def save_clusters(self, clusters: Any, path: str) -> None:
        """Save cluster labels."""
        import numpy as np

        full_path = self.base_path / path
        full_path.parent.mkdir(parents=True, exist_ok=True)

        np.save(full_path, clusters)


class ResultStorageStage(PipelineStageInterface):
    """
    Pipeline stage for storing results.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        base_path = config.get('base_path', 'results')
        self.repository = FileSystemRepository(base_path)

        # Generate timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = config.get('run_dir', f"run_{self.timestamp}")

        self.save_embeddings = config.get('save_embeddings', True)
        self.save_clusters = config.get('save_clusters', True)
        self.save_visualizations = config.get('save_visualizations', True)

    def validate(self, context: PipelineContext) -> bool:
        """Validate that results exist."""
        return context.clusters is not None

    def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Save all results.

        Args:
            context: Current pipeline context

        Returns:
            Updated context with results metadata
        """
        self.notify_observers("result_storage_started", {})

        try:
            results = {}

            # Save metrics
            metrics = self._compile_metrics(context)
            metrics_path = f"{self.run_dir}/metrics.json"
            self.repository.save_metrics(metrics, metrics_path)
            results['metrics_path'] = metrics_path

            # Save embeddings
            if self.save_embeddings and context.embeddings is not None:
                embeddings_path = f"{self.run_dir}/embeddings.npy"
                self.repository.save_embeddings(context.embeddings, embeddings_path)
                results['embeddings_path'] = embeddings_path

            # Save clusters
            if self.save_clusters and context.clusters is not None:
                clusters_path = f"{self.run_dir}/clusters.npy"
                self.repository.save_clusters(context.clusters, clusters_path)
                results['clusters_path'] = clusters_path

            # Generate and save visualizations
            if self.save_visualizations:
                viz_paths = self._generate_visualizations(context)
                results['visualization_paths'] = viz_paths

            self.notify_observers("result_storage_completed", {
                "results": results
            })

            context.results = results
            context.metadata['storage_paths'] = results

        except Exception as e:
            self.notify_observers("result_storage_failed", {
                "error": str(e)
            })
            raise

        return context

    def _compile_metrics(self, context: PipelineContext) -> Dict[str, Any]:
        """Compile all metrics from the pipeline."""
        metrics = {
            'timestamp': self.timestamp,
            'run_dir': self.run_dir,
            'pipeline_metadata': context.metadata
        }

        return metrics

    def _generate_visualizations(self, context: PipelineContext) -> List[str]:
        """
        Generate visualizations.
        This delegates to the visualization module.
        """
        from src.visualization.generator import VisualizationGenerator

        viz_generator = VisualizationGenerator(
            output_dir=str(self.repository.base_path / self.run_dir / "visualizations")
        )

        viz_paths = []

        # Generate graph visualization
        if context.graph is not None:
            path = viz_generator.visualize_graph(context.graph)
            if path:
                viz_paths.append(path)

        # Generate embedding visualization
        if context.embeddings is not None and context.clusters is not None:
            path = viz_generator.visualize_embeddings(
                context.embeddings,
                context.clusters
            )
            if path:
                viz_paths.append(path)

        # Generate cluster distribution
        if context.clusters is not None:
            path = viz_generator.visualize_cluster_distribution(context.clusters)
            if path:
                viz_paths.append(path)

        return viz_paths
