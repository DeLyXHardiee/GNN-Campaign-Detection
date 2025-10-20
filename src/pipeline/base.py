"""
Core pipeline interfaces and base classes.

This module defines the abstract base classes for the pipeline architecture
using the Pipeline and Strategy design patterns.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class PipelineStage(Enum):
    """Enumeration of pipeline stages."""
    DATA_LOADING = "data_loading"
    MISP_CONVERSION = "misp_conversion"
    GRAPH_GENERATION = "graph_generation"
    GNN_PROCESSING = "gnn_processing"
    CLUSTERING = "clustering"
    RESULT_STORAGE = "result_storage"


@dataclass
class PipelineContext:
    """
    Shared context passed between pipeline stages.
    Acts as a data container for intermediate results.
    """
    raw_data: Optional[Any] = None
    misp_objects: Optional[List[Any]] = None
    graph: Optional[Any] = None
    embeddings: Optional[Any] = None
    clusters: Optional[Any] = None
    results: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class PipelineStageInterface(ABC):
    """
    Abstract base class for pipeline stages.
    Each stage processes the context and returns an updated context.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.observers: List['PipelineObserver'] = []

    def add_observer(self, observer: 'PipelineObserver') -> None:
        """Add an observer for monitoring."""
        self.observers.append(observer)

    def notify_observers(self, event: str, data: Dict[str, Any]) -> None:
        """Notify all observers of an event."""
        for observer in self.observers:
            observer.update(event, data)

    @abstractmethod
    def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Execute the pipeline stage.

        Args:
            context: The current pipeline context

        Returns:
            Updated pipeline context
        """
        pass

    @abstractmethod
    def validate(self, context: PipelineContext) -> bool:
        """
        Validate that the context has the required data for this stage.

        Args:
            context: The current pipeline context

        Returns:
            True if validation passes, False otherwise
        """
        pass

    def get_stage_name(self) -> str:
        """Get the name of this stage."""
        return self.__class__.__name__


class PipelineObserver(ABC):
    """
    Observer interface for monitoring pipeline execution.
    Implements the Observer pattern.
    """

    @abstractmethod
    def update(self, event: str, data: Dict[str, Any]) -> None:
        """
        Receive update from observed pipeline stage.

        Args:
            event: Event name/type
            data: Event data
        """
        pass


class Pipeline:
    """
    Main pipeline orchestrator.
    Executes stages sequentially and manages the context.
    """

    def __init__(self, stages: List[PipelineStageInterface]):
        self.stages = stages
        self.observers: List[PipelineObserver] = []

    def add_observer(self, observer: PipelineObserver) -> None:
        """Add a global pipeline observer."""
        self.observers.append(observer)
        # Propagate to all stages
        for stage in self.stages:
            stage.add_observer(observer)

    def execute(self) -> PipelineContext:
        """
        Execute the pipeline end-to-end.

        Returns:
            Final pipeline context with all results
        """
        context = PipelineContext()

        self._notify_observers("pipeline_started", {})

        for i, stage in enumerate(self.stages):
            stage_name = stage.get_stage_name()

            self._notify_observers(
                "stage_started",
                {"stage": stage_name, "index": i}
            )

            # Validate prerequisites
            if not stage.validate(context):
                error_msg = f"Validation failed for stage: {stage_name}"
                self._notify_observers(
                    "stage_failed",
                    {"stage": stage_name, "error": error_msg}
                )
                raise ValueError(error_msg)

            # Execute stage
            try:
                context = stage.execute(context)
                self._notify_observers(
                    "stage_completed",
                    {"stage": stage_name, "index": i}
                )
            except Exception as e:
                self._notify_observers(
                    "stage_failed",
                    {"stage": stage_name, "error": str(e)}
                )
                raise

        self._notify_observers("pipeline_completed", {})

        return context

    def _notify_observers(self, event: str, data: Dict[str, Any]) -> None:
        """Notify all global observers."""
        for observer in self.observers:
            observer.update(event, data)
