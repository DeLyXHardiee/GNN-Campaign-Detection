# Architecture Documentation

## Overview

This project implements a modular, extensible pipeline for GNN-based campaign detection research. The architecture emphasizes scalability, maintainability, and extensibility through the use of well-established design patterns.

## Design Patterns

### 1. Pipeline Pattern

The core architecture uses the **Pipeline Pattern** to organize sequential processing stages:

```
Data Loading → MISP Conversion → Graph Generation → GNN Processing → Clustering → Storage
```

**Benefits:**
- Clear separation of concerns
- Easy to add/remove/reorder stages
- Each stage is independently testable
- Consistent interface across all stages

**Implementation:**
- `PipelineStageInterface`: Base class for all stages
- `Pipeline`: Orchestrates stage execution
- `PipelineContext`: Carries data between stages

### 2. Strategy Pattern

Used for interchangeable algorithms:

**Examples:**
- `GNNModel`: Different GNN architectures (Baseline, Node2Vec, custom)
- `ClusteringStrategy`: Different clustering algorithms (K-Means, DBSCAN, Hierarchical)
- `DataLoader`: Different data sources (CSV, multi-file, databases)
- `MISPConverter`: Different conversion strategies

**Benefits:**
- Easy to switch algorithms without changing pipeline code
- Supports A/B testing and algorithm comparison
- Facilitates research experimentation

### 3. Factory Pattern

Used for object creation:

**Implementations:**
- `DataLoaderFactory`: Creates appropriate data loaders
- `MISPConverterFactory`: Creates MISP converters
- `GraphBuilderFactory`: Creates graph builders
- `GNNModelFactory`: Creates GNN models
- `ClusteringFactory`: Creates clustering algorithms

**Benefits:**
- Centralizes object creation logic
- Supports configuration-driven instantiation
- Easy to add new types without modifying client code

### 4. Observer Pattern

Used for monitoring and logging:

**Implementations:**
- `PipelineObserver`: Base observer interface
- `LoggerObserver`: Logs pipeline events
- `MetricsCollector`: Collects performance metrics

**Benefits:**
- Decouples monitoring from business logic
- Multiple observers can monitor simultaneously
- Easy to add custom monitoring without modifying pipeline

### 5. Builder Pattern

Used for complex object construction:

**Implementations:**
- `PipelineBuilder`: Constructs complete pipelines
- `GraphBuilder`: Constructs complex graph structures
- `PipelineConfig`: Builds configuration from files

**Benefits:**
- Separates construction from representation
- Supports step-by-step configuration
- Fluent interface for readability

### 6. Repository Pattern

Used for data persistence:

**Implementation:**
- `ResultRepository`: Abstract storage interface
- `FileSystemRepository`: File-based storage
- (Can add: `DatabaseRepository`, `S3Repository`, etc.)

**Benefits:**
- Abstracts storage mechanism
- Easy to switch between storage backends
- Facilitates testing with mock repositories

## Component Architecture

### Core Components

```
src/
├── pipeline/          # Pipeline orchestration
│   ├── base.py       # Base classes and interfaces
│   ├── builder.py    # Pipeline builder
│   └── __init__.py
│
├── data/             # Data loading
│   ├── loader.py     # Data loader implementations
│   └── __init__.py
│
├── misp/             # MISP conversion
│   ├── converter.py  # MISP converter implementations
│   └── __init__.py
│
├── graph/            # Graph generation
│   ├── generator.py  # Graph builder implementations
│   └── __init__.py
│
├── models/           # GNN models
│   ├── processor.py  # GNN model implementations
│   └── __init__.py
│
├── clustering/       # Clustering algorithms
│   ├── processor.py  # Clustering implementations
│   └── __init__.py
│
├── storage/          # Result storage
│   ├── saver.py      # Storage implementations
│   └── __init__.py
│
├── visualization/    # Visualization
│   ├── generator.py  # Visualization generators
│   └── __init__.py
│
└── utils/           # Utilities
    ├── logging.py   # Logging and monitoring
    └── __init__.py
```

## Extension Points

### Adding a New Pipeline Stage

1. Inherit from `PipelineStageInterface`
2. Implement `execute()` and `validate()` methods
3. Add to pipeline in `PipelineBuilder`

```python
class NewStage(PipelineStageInterface):
    def validate(self, context: PipelineContext) -> bool:
        # Check prerequisites
        return True

    def execute(self, context: PipelineContext) -> PipelineContext:
        # Process data
        return context
```

### Adding a New GNN Model

1. Inherit from `GNNModel`
2. Implement `train()` and `generate_embeddings()`
3. Register in `GNNModelFactory`

```python
class CustomGNN(GNNModel):
    def train(self, graph, **kwargs):
        # Training logic
        pass

    def generate_embeddings(self, graph):
        # Embedding generation
        return embeddings
```

### Adding a New Clustering Algorithm

1. Inherit from `ClusteringStrategy`
2. Implement `fit_predict()` and `get_metrics()`
3. Register in `ClusteringFactory`

## Configuration System

The pipeline uses YAML-based configuration for flexibility:

```yaml
# config/default_config.yaml
data_loading:
  loader_type: "csv"
  data_source: "data/input.csv"

gnn_processing:
  model_type: "baseline"
  model_kwargs:
    embedding_dim: 64
```

This allows:
- Easy experimentation with different configurations
- Reproducibility through config versioning
- No code changes for parameter tuning

## Testing Strategy

Each component is independently testable:

1. **Unit Tests**: Test individual classes and functions
2. **Integration Tests**: Test stage interactions
3. **Pipeline Tests**: Test end-to-end execution

Mock objects are used to isolate components during testing.

## Scalability Considerations

1. **Parallelization**: Stages could be parallelized where independent
2. **Distributed Processing**: Could extend to distributed graphs
3. **Streaming**: Could add streaming data support
4. **Caching**: Could add intermediate result caching

## Future Enhancements

1. **Checkpoint/Resume**: Save pipeline state for long runs
2. **Distributed Execution**: Support for cluster computing
3. **Real-time Processing**: Streaming data support
4. **Model Registry**: Centralized model management
5. **Experiment Tracking**: Integration with MLflow/Weights & Biases
6. **API Layer**: REST API for pipeline execution
