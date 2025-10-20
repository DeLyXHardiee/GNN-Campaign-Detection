"""
Unit tests for MISP converter.
"""

import pytest
import pandas as pd

from src.misp.converter import DefaultMISPConverter, MISPObject


def test_misp_object_creation():
    """Test MISP object creation."""
    obj = MISPObject(
        object_type='test',
        attributes={'key': 'value'}
    )

    assert obj.object_type == 'test'
    assert obj.attributes['key'] == 'value'
    assert obj.uuid is not None


def test_default_converter():
    """Test default MISP converter."""
    data = pd.DataFrame({
        'type': ['event', 'indicator'],
        'value': ['val1', 'val2']
    })

    converter = DefaultMISPConverter(object_type_column='type')
    objects = converter.convert(data)

    assert len(objects) == 2
    assert all(isinstance(obj, MISPObject) for obj in objects)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
