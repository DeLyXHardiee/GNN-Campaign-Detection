"""
Advanced example: Custom pipeline with specific stages.
"""

from src.pipeline import PipelineBuilder, PipelineConfig, Pipeline
from src.data.loader import DataLoaderStage
from src.misp.converter import MISPConverterStage
from src.graph.generator import GraphGeneratorStage
from src.models.processor import GNNProcessorStage
from src.clustering.processor import ClusteringProcessorStage
from src.storage.saver import ResultStorageStage
from src.utils.logging import LoggerObserver


def main():
    # Load base configuration
    config = PipelineConfig.from_file('config/advanced_config.yaml')

    # Manually create stages with custom configurations
    stages = [
        DataLoaderStage({
            'loader_type': 'csv',
            'data_source': 'data/campaign_data.csv'
        }),
        MISPConverterStage({
            'converter_type': 'default',
            'converter_kwargs': {
                'object_type_column': 'entity_type'
            }
        }),
        GraphGeneratorStage({
            'builder_type': 'networkx',
            'builder_kwargs': {
                'directed': True,
                'include_node_attributes': True
            }
        }),
        GNNProcessorStage({
            'model_type': 'baseline',
            'model_kwargs': {
                'embedding_dim': 128,
                'num_layers': 3
            },
            'training': {
                'epochs': 200
            }
        }),
        ClusteringProcessorStage({
            'algorithm': 'hierarchical',
            'algorithm_kwargs': {
                'n_clusters': 10,
                'linkage': 'ward'
            }
        }),
        ResultStorageStage({
            'base_path': 'results',
            'run_dir': 'custom_run',
            'save_embeddings': True,
            'save_clusters': True,
            'save_visualizations': True
        })
    ]

    # Create pipeline manually
    pipeline = Pipeline(stages)
    pipeline.add_observer(LoggerObserver())

    # Execute
    print("Running custom pipeline...")
    context = pipeline.execute()
    print(f"Results: {context.results}")


if __name__ == "__main__":
    main()
