"""
Pipeline builder and configuration.

Implements the Builder pattern for constructing pipelines.
"""

from typing import Dict, Any, List, Optional
import yaml
from pathlib import Path

from .base import Pipeline, PipelineStageInterface, PipelineObserver


class PipelineConfig:
    """
    Configuration container for the pipeline.
    Loads and validates configuration from files or dictionaries.
    """

    def __init__(self, config_dict: Dict[str, Any]):
        self.config = config_dict
        self._validate()

    @classmethod
    def from_file(cls, config_path: str) -> 'PipelineConfig':
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PipelineConfig':
        """Create configuration from dictionary."""
        return cls(config_dict)

    def _validate(self) -> None:
        """Validate configuration structure."""
        required_sections = [
            'data_loading',
            'misp_conversion',
            'graph_generation',
            'gnn_processing',
            'clustering',
            'result_storage'
        ]

        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required config section: {section}")

    def get_stage_config(self, stage_name: str) -> Dict[str, Any]:
        """Get configuration for a specific stage."""
        return self.config.get(stage_name, {})

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self.config.get(key, default)


class PipelineBuilder:
    """
    Builder for constructing pipeline instances.
    Implements the Builder pattern.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.stages: List[PipelineStageInterface] = []
        self.observers: List[PipelineObserver] = []

    def add_stage(self, stage: PipelineStageInterface) -> 'PipelineBuilder':
        """Add a stage to the pipeline."""
        self.stages.append(stage)
        return self

    def add_observer(self, observer: PipelineObserver) -> 'PipelineBuilder':
        """Add an observer to the pipeline."""
        self.observers.append(observer)
        return self

    def build(self) -> Pipeline:
        """
        Build the complete pipeline with all configured stages.

        Returns:
            Configured Pipeline instance
        """
        # Import stage implementations
        from src.data.loader import DataLoaderStage
        from src.misp.converter import MISPConverterStage
        from src.graph.generator import GraphGeneratorStage
        from src.models.processor import GNNProcessorStage
        from src.clustering.processor import ClusteringProcessorStage
        from src.storage.saver import ResultStorageStage

        # Build stages in order
        if not self.stages:  # Auto-build from config
            self.stages = [
                DataLoaderStage(self.config.get_stage_config('data_loading')),
                MISPConverterStage(self.config.get_stage_config('misp_conversion')),
                GraphGeneratorStage(self.config.get_stage_config('graph_generation')),
                GNNProcessorStage(self.config.get_stage_config('gnn_processing')),
                ClusteringProcessorStage(self.config.get_stage_config('clustering')),
                ResultStorageStage(self.config.get_stage_config('result_storage'))
            ]

        # Create pipeline
        pipeline = Pipeline(self.stages)

        # Add observers
        for observer in self.observers:
            pipeline.add_observer(observer)

        # Add default logger if no observers
        if not self.observers:
            from src.utils.logging import LoggerObserver
            pipeline.add_observer(LoggerObserver())

        return pipeline
