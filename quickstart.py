"""
Quick start script to set up the project and run a demo.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np


def create_sample_data():
    """Create sample data for demonstration."""
    print("Creating sample data...")

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Generate sample data
    np.random.seed(42)

    n_samples = 100
    sample_data = {
        'type': np.random.choice(['event', 'indicator', 'attribute'], n_samples),
        'value': [f'value_{i}' for i in range(n_samples)],
        'category': np.random.choice(['malware', 'network', 'artifact'], n_samples),
        'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='H'),
        'severity': np.random.choice(['low', 'medium', 'high'], n_samples)
    }

    df = pd.DataFrame(sample_data)
    output_path = data_dir / "sample_data.csv"
    df.to_csv(output_path, index=False)

    print(f"Sample data created: {output_path}")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    return output_path


def update_config_for_demo(data_path):
    """Update configuration to use sample data."""
    print("\nUpdating configuration...")

    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)

    config_content = f"""# Demo configuration

data_loading:
  loader_type: "csv"
  data_source: "{str(data_path).replace(chr(92), '/')}"

misp_conversion:
  converter_type: "default"
  converter_kwargs:
    object_type_column: "type"

graph_generation:
  builder_type: "networkx"
  builder_kwargs:
    directed: true
    include_node_attributes: true
    include_edge_attributes: true

gnn_processing:
  model_type: "baseline"
  model_kwargs:
    embedding_dim: 32
    num_layers: 2
    learning_rate: 0.01
  training:
    epochs: 50

clustering:
  algorithm: "kmeans"
  algorithm_kwargs:
    n_clusters: 3
    random_state: 42

result_storage:
  base_path: "results"
  save_embeddings: true
  save_clusters: true
  save_visualizations: true
"""

    config_path = config_dir / "demo_config.yaml"
    with open(config_path, 'w') as f:
        f.write(config_content)

    print(f"Configuration updated: {config_path}")

    return config_path


def run_demo_pipeline(config_path):
    """Run the pipeline with demo configuration."""
    print("\n" + "="*60)
    print("RUNNING PIPELINE DEMO")
    print("="*60 + "\n")

    try:
        from src.pipeline import PipelineBuilder, PipelineConfig
        from src.utils.logging import LoggerObserver, MetricsCollector

        # Load configuration
        config = PipelineConfig.from_file(str(config_path))

        # Create observers
        logger = LoggerObserver(log_file='logs/demo.log', verbose=True)
        metrics_collector = MetricsCollector()

        # Build pipeline
        pipeline = (
            PipelineBuilder(config)
            .add_observer(logger)
            .add_observer(metrics_collector)
            .build()
        )

        # Execute pipeline
        context = pipeline.execute()

        # Print results
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("="*60)

        print("\nResults Summary:")
        print(f"  Results directory: {context.results}")

        print("\nMetrics:")
        for key, value in metrics_collector.get_metrics().items():
            print(f"  {key}: {value}")

        print("\nPipeline Metadata:")
        for key, value in context.metadata.items():
            print(f"  {key}: {value}")

        return True

    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to run the demo."""
    print("="*60)
    print("GNN CAMPAIGN DETECTION - QUICK START DEMO")
    print("="*60 + "\n")

    # Step 1: Create sample data
    try:
        data_path = create_sample_data()
    except Exception as e:
        print(f"❌ Failed to create sample data: {e}")
        return

    # Step 2: Update configuration
    try:
        config_path = update_config_for_demo(data_path)
    except Exception as e:
        print(f"❌ Failed to update configuration: {e}")
        return

    # Step 3: Run pipeline
    success = run_demo_pipeline(config_path)

    if success:
        print("\n✅ Demo completed successfully!")
        print("\nNext steps:")
        print("  1. Check the results/ directory for outputs")
        print("  2. View logs in logs/demo.log")
        print("  3. Read docs/USAGE.md for more information")
        print("  4. Modify config/demo_config.yaml to experiment")
    else:
        print("\n❌ Demo failed. Check the error messages above.")


if __name__ == "__main__":
    main()
