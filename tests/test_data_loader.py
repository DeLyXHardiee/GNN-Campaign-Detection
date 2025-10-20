"""
Unit tests for data loading stage.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch
from pathlib import Path

from src.data.loader import CSVDataLoader, DataLoaderStage
from src.pipeline import PipelineContext


def test_csv_data_loader():
    """Test CSV data loader."""
    # This would require an actual CSV file or mocking
    # For architecture demonstration, we show the test structure
    pass


def test_data_loader_stage_validation():
    """Test data loader stage validation."""
    config = {
        'loader_type': 'csv',
        'data_source': 'test.csv'
    }

    stage = DataLoaderStage(config)
    context = PipelineContext()

    # Should always validate at start
    assert stage.validate(context)


def test_data_loader_stage_config_validation():
    """Test that data loader stage requires data_source."""
    config = {
        'loader_type': 'csv'
        # Missing data_source
    }

    with pytest.raises(ValueError):
        DataLoaderStage(config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
