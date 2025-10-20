"""
Unit tests for the pipeline architecture.
"""

import pytest
from unittest.mock import Mock, MagicMock
import numpy as np

from src.pipeline import Pipeline, PipelineContext, PipelineStageInterface


class MockStage(PipelineStageInterface):
    """Mock stage for testing."""

    def __init__(self, config):
        super().__init__(config)
        self.executed = False

    def validate(self, context):
        return True

    def execute(self, context):
        self.executed = True
        return context


def test_pipeline_context_initialization():
    """Test PipelineContext initialization."""
    context = PipelineContext()
    assert context.raw_data is None
    assert context.misp_objects is None
    assert context.graph is None
    assert context.embeddings is None
    assert context.clusters is None
    assert context.metadata is not None
    assert isinstance(context.metadata, dict)


def test_pipeline_stage_execution():
    """Test basic pipeline stage execution."""
    stage = MockStage({})
    context = PipelineContext()

    assert not stage.executed
    result_context = stage.execute(context)
    assert stage.executed
    assert result_context is context


def test_pipeline_execution():
    """Test complete pipeline execution."""
    stage1 = MockStage({'name': 'stage1'})
    stage2 = MockStage({'name': 'stage2'})

    pipeline = Pipeline([stage1, stage2])
    context = pipeline.execute()

    assert stage1.executed
    assert stage2.executed
    assert isinstance(context, PipelineContext)


def test_pipeline_observer():
    """Test pipeline observer pattern."""
    from src.utils.logging import MetricsCollector

    stage = MockStage({})
    collector = MetricsCollector()

    pipeline = Pipeline([stage])
    pipeline.add_observer(collector)

    pipeline.execute()

    metrics = collector.get_metrics()
    assert 'total_duration_seconds' in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
