"""
Example: Extending the pipeline with a custom MISP converter.
"""

import pandas as pd
from typing import List

from src.misp.converter import MISPConverter, MISPObject, MISPRelationship
from src.pipeline import PipelineBuilder, PipelineConfig


class CustomMISPConverter(MISPConverter):
    """
    Example custom MISP converter for domain-specific data.
    """

    def __init__(self, source_column: str, target_column: str):
        self.source_column = source_column
        self.target_column = target_column

    def convert(self, data: pd.DataFrame) -> List[MISPObject]:
        """Convert data rows to MISP objects."""
        objects = {}

        for _, row in data.iterrows():
            # Create object for source
            source_id = row[self.source_column]
            if source_id not in objects:
                objects[source_id] = MISPObject(
                    object_type='source',
                    attributes={'id': source_id}
                )

            # Create object for target
            target_id = row[self.target_column]
            if target_id not in objects:
                objects[target_id] = MISPObject(
                    object_type='target',
                    attributes={'id': target_id}
                )

        return list(objects.values())

    def extract_relationships(
        self,
        objects: List[MISPObject]
    ) -> List[MISPRelationship]:
        """Extract relationships between objects."""
        # Custom relationship extraction logic
        relationships = []

        # Example: Create relationships based on shared attributes
        for i, obj1 in enumerate(objects):
            for obj2 in objects[i+1:]:
                # Define your relationship criteria here
                if self._should_relate(obj1, obj2):
                    rel = MISPRelationship(
                        source_uuid=obj1.uuid,
                        target_uuid=obj2.uuid,
                        relationship_type='related-to'
                    )
                    relationships.append(rel)

        return relationships

    def _should_relate(self, obj1: MISPObject, obj2: MISPObject) -> bool:
        """Determine if two objects should be related."""
        # Implement your logic here
        return False  # Placeholder


def main():
    """Run pipeline with custom converter."""

    # Note: To use the custom converter in the pipeline,
    # you would need to either:
    # 1. Register it in the MISPConverterFactory
    # 2. Manually create stages with your custom converter

    # Example of manual approach:
    from src.misp.converter import MISPConverterStage
    from src.pipeline import Pipeline
    from src.data.loader import DataLoaderStage

    # Override the converter in the stage
    converter_stage = MISPConverterStage({
        'converter_type': 'default'
    })
    # Replace with custom converter
    converter_stage.converter = CustomMISPConverter(
        source_column='source_ip',
        target_column='dest_ip'
    )

    print("Custom converter example - see code for implementation details")


if __name__ == "__main__":
    main()
