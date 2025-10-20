"""
Basic example: Run the pipeline with default configuration.
"""

from src.pipeline import PipelineBuilder, PipelineConfig
from src.utils.logging import LoggerObserver, MetricsCollector


def main():
    # Load configuration
    config = PipelineConfig.from_file('config/default_config.yaml')

    # Create observers
    logger = LoggerObserver(log_file='logs/pipeline.log', verbose=True)
    metrics_collector = MetricsCollector()

    # Build pipeline
    pipeline = (
        PipelineBuilder(config)
        .add_observer(logger)
        .add_observer(metrics_collector)
        .build()
    )

    # Execute pipeline
    print("Starting pipeline execution...")
    try:
        context = pipeline.execute()
        print("\nPipeline completed successfully!")

        # Print results
        print(f"\nResults saved to: {context.results}")

        # Print collected metrics
        print("\nPipeline Metrics:")
        for key, value in metrics_collector.get_metrics().items():
            print(f"  {key}: {value}")

    except Exception as e:
        print(f"\nPipeline failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
