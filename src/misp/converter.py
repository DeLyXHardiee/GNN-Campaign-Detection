"""
MISP object conversion stage.

Converts raw data to MISP objects using configurable converters.
Implements the Strategy and Factory patterns.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import pandas as pd

from src.pipeline.base import PipelineStageInterface, PipelineContext


class MISPObject:
    """
    Simplified MISP object representation.
    In production, this would use pymisp.MISPObject.
    """

    def __init__(
        self,
        object_type: str,
        attributes: Dict[str, Any],
        uuid: Optional[str] = None
    ):
        self.object_type = object_type
        self.attributes = attributes
        self.uuid = uuid or self._generate_uuid()
        self.relationships: List['MISPRelationship'] = []

    def _generate_uuid(self) -> str:
        """Generate a UUID for this object."""
        import uuid
        return str(uuid.uuid4())

    def add_relationship(self, relationship: 'MISPRelationship') -> None:
        """Add a relationship to another MISP object."""
        self.relationships.append(relationship)

    def __repr__(self) -> str:
        return f"MISPObject(type={self.object_type}, uuid={self.uuid})"


class MISPRelationship:
    """Represents a relationship between MISP objects."""

    def __init__(
        self,
        source_uuid: str,
        target_uuid: str,
        relationship_type: str
    ):
        self.source_uuid = source_uuid
        self.target_uuid = target_uuid
        self.relationship_type = relationship_type

    def __repr__(self) -> str:
        return f"MISPRelationship({self.relationship_type}: {self.source_uuid} -> {self.target_uuid})"


class MISPConverter(ABC):
    """
    Abstract base class for MISP converters (Strategy pattern).
    """

    @abstractmethod
    def convert(self, data: pd.DataFrame) -> List[MISPObject]:
        """
        Convert DataFrame to MISP objects.

        Args:
            data: Input DataFrame

        Returns:
            List of MISP objects
        """
        pass

    @abstractmethod
    def extract_relationships(
        self,
        objects: List[MISPObject]
    ) -> List[MISPRelationship]:
        """
        Extract relationships between MISP objects.

        Args:
            objects: List of MISP objects

        Returns:
            List of relationships
        """
        pass


class DefaultMISPConverter(MISPConverter):
    """
    Default MISP converter implementation.
    This is a baseline that should be extended for specific data formats.
    """

    def __init__(
        self,
        object_type_column: str = 'type',
        attribute_columns: Optional[List[str]] = None
    ):
        self.object_type_column = object_type_column
        self.attribute_columns = attribute_columns

    def convert(self, data: pd.DataFrame) -> List[MISPObject]:
        """Convert each row to a MISP object."""
        objects = []

        for _, row in data.iterrows():
            # Determine object type
            obj_type = row.get(self.object_type_column, 'unknown')

            # Extract attributes
            if self.attribute_columns:
                attributes = {col: row[col] for col in self.attribute_columns if col in row}
            else:
                attributes = row.to_dict()

            # Create MISP object
            obj = MISPObject(object_type=str(obj_type), attributes=attributes)
            objects.append(obj)

        return objects

    def extract_relationships(
        self,
        objects: List[MISPObject]
    ) -> List[MISPRelationship]:
        """
        Extract relationships (baseline implementation).
        Override this method for specific relationship extraction logic.
        """
        relationships = []
        # Baseline: no relationships extracted
        # Subclasses should implement domain-specific logic
        return relationships


class MISPConverterFactory:
    """
    Factory for creating MISP converters.
    Implements the Factory pattern.
    """

    @staticmethod
    def create(converter_type: str, **kwargs) -> MISPConverter:
        """
        Create a MISP converter instance.

        Args:
            converter_type: Type of converter ('default', custom types)
            **kwargs: Additional arguments for the converter

        Returns:
            MISPConverter instance
        """
        if converter_type == 'default':
            return DefaultMISPConverter(**kwargs)
        else:
            raise ValueError(f"Unknown converter type: {converter_type}")


class MISPConverterStage(PipelineStageInterface):
    """
    Pipeline stage for converting data to MISP objects.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        converter_type = config.get('converter_type', 'default')
        converter_kwargs = config.get('converter_kwargs', {})

        self.converter = MISPConverterFactory.create(
            converter_type,
            **converter_kwargs
        )

    def validate(self, context: PipelineContext) -> bool:
        """Validate that raw data exists."""
        return context.raw_data is not None

    def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Convert data to MISP objects and update context.

        Args:
            context: Current pipeline context

        Returns:
            Updated context with misp_objects
        """
        self.notify_observers("misp_conversion_started", {})

        try:
            # Convert to MISP objects
            misp_objects = self.converter.convert(context.raw_data)

            # Extract relationships
            relationships = self.converter.extract_relationships(misp_objects)

            # Add relationships to objects
            for rel in relationships:
                source = next(
                    (obj for obj in misp_objects if obj.uuid == rel.source_uuid),
                    None
                )
                if source:
                    source.add_relationship(rel)

            self.notify_observers("misp_conversion_completed", {
                "objects_created": len(misp_objects),
                "relationships_created": len(relationships)
            })

            context.misp_objects = misp_objects
            context.metadata['misp_object_count'] = len(misp_objects)
            context.metadata['misp_relationship_count'] = len(relationships)

        except Exception as e:
            self.notify_observers("misp_conversion_failed", {
                "error": str(e)
            })
            raise

        return context
