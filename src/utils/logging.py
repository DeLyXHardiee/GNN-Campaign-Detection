"""
Logging utilities.

Implements the Observer pattern for pipeline monitoring.
"""

from typing import Any, Dict
from datetime import datetime
from pathlib import Path

from src.pipeline.base import PipelineObserver


class LoggerObserver(PipelineObserver):
    """
    Logger observer for pipeline events.
    Uses loguru for structured logging.
    """

    def __init__(self, log_file: str = "logs/pipeline.log", verbose: bool = True):
        self.verbose = verbose
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # In production, initialize loguru logger
        # from loguru import logger
        # logger.add(log_file, rotation="500 MB")
        # self.logger = logger

    def update(self, event: str, data: Dict[str, Any]) -> None:
        """
        Log pipeline events.

        Args:
            event: Event name
            data: Event data
        """
        timestamp = datetime.now().isoformat()
        log_message = self._format_message(event, data)

        if self.verbose:
            print(f"[{timestamp}] {log_message}")

        # In production: self.logger.info(log_message, **data)

        # Write to file
        with open(self.log_file, 'a') as f:
            f.write(f"[{timestamp}] {log_message}\n")

    def _format_message(self, event: str, data: Dict[str, Any]) -> str:
        """Format log message based on event type."""
        if event == "pipeline_started":
            return "Pipeline execution started"

        elif event == "pipeline_completed":
            return "Pipeline execution completed successfully"

        elif event == "stage_started":
            return f"Stage started: {data.get('stage', 'unknown')}"

        elif event == "stage_completed":
            return f"Stage completed: {data.get('stage', 'unknown')}"

        elif event == "stage_failed":
            return f"Stage failed: {data.get('stage', 'unknown')} - {data.get('error', 'unknown error')}"

        elif event == "data_loading_started":
            return f"Loading data from: {data.get('source', 'unknown')}"

        elif event == "data_loading_completed":
            return f"Data loaded: {data.get('rows', 0)} rows, {data.get('columns', 0)} columns"

        elif event == "misp_conversion_completed":
            return f"MISP conversion completed: {data.get('objects_created', 0)} objects, {data.get('relationships_created', 0)} relationships"

        elif event == "graph_generation_completed":
            return f"Graph generated: {data.get('nodes', 0)} nodes, {data.get('edges', 0)} edges"

        elif event == "gnn_training_started":
            return "GNN training started"

        elif event == "gnn_training_completed":
            return "GNN training completed"

        elif event == "embedding_generation_completed":
            return f"Embeddings generated: shape {data.get('embedding_shape', 'unknown')}"

        elif event == "clustering_completed":
            return f"Clustering completed: {data.get('unique_clusters', 0)} clusters"

        elif event == "result_storage_completed":
            return "Results saved successfully"

        else:
            return f"{event}: {data}"


class MetricsCollector(PipelineObserver):
    """
    Collects metrics throughout pipeline execution.
    """

    def __init__(self):
        self.metrics = {}
        self.start_time = None
        self.end_time = None

    def update(self, event: str, data: Dict[str, Any]) -> None:
        """Collect metrics from pipeline events."""
        if event == "pipeline_started":
            self.start_time = datetime.now()

        elif event == "pipeline_completed":
            self.end_time = datetime.now()
            if self.start_time:
                duration = (self.end_time - self.start_time).total_seconds()
                self.metrics['total_duration_seconds'] = duration

        elif event == "data_loading_completed":
            self.metrics['data_rows'] = data.get('rows', 0)
            self.metrics['data_columns'] = data.get('columns', 0)

        elif event == "misp_conversion_completed":
            self.metrics['misp_objects'] = data.get('objects_created', 0)
            self.metrics['misp_relationships'] = data.get('relationships_created', 0)

        elif event == "graph_generation_completed":
            self.metrics['graph_nodes'] = data.get('nodes', 0)
            self.metrics['graph_edges'] = data.get('edges', 0)

        elif event == "embedding_generation_completed":
            self.metrics['embedding_shape'] = str(data.get('embedding_shape', 'unknown'))

        elif event == "clustering_completed":
            self.metrics['n_clusters'] = data.get('unique_clusters', 0)
            self.metrics['clustering_metrics'] = data.get('metrics', {})

    def get_metrics(self) -> Dict[str, Any]:
        """Get collected metrics."""
        return self.metrics.copy()
