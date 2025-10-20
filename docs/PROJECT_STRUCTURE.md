# Project Structure Overview

```
GNN-Campaign-Detection/
│
├── README.md                    # Project overview and quick start
├── CONTRIBUTING.md              # Contribution guidelines
├── requirements.txt             # Python dependencies
├── .gitignore                  # Git ignore patterns
├── quickstart.py               # Demo script to get started quickly
│
├── config/                     # Configuration files
│   ├── default_config.yaml     # Default pipeline configuration
│   └── advanced_config.yaml    # Advanced configuration example
│
├── src/                        # Source code
│   ├── __init__.py
│   │
│   ├── pipeline/               # Core pipeline architecture
│   │   ├── __init__.py
│   │   ├── base.py            # Base classes (Pipeline, PipelineStage, etc.)
│   │   └── builder.py         # PipelineBuilder and PipelineConfig
│   │
│   ├── data/                   # Data loading stage
│   │   ├── __init__.py
│   │   └── loader.py          # DataLoader implementations
│   │
│   ├── misp/                   # MISP conversion stage
│   │   ├── __init__.py
│   │   └── converter.py       # MISPConverter implementations
│   │
│   ├── graph/                  # Graph generation stage
│   │   ├── __init__.py
│   │   └── generator.py       # GraphBuilder implementations
│   │
│   ├── models/                 # GNN processing stage
│   │   ├── __init__.py
│   │   └── processor.py       # GNNModel implementations
│   │
│   ├── clustering/             # Clustering stage
│   │   ├── __init__.py
│   │   └── processor.py       # ClusteringStrategy implementations
│   │
│   ├── storage/                # Result storage stage
│   │   ├── __init__.py
│   │   └── saver.py           # ResultRepository implementations
│   │
│   ├── visualization/          # Visualization components
│   │   ├── __init__.py
│   │   └── generator.py       # VisualizationGenerator
│   │
│   └── utils/                  # Utility modules
│       ├── __init__.py
│       └── logging.py         # Logging and monitoring (Observers)
│
├── examples/                   # Example scripts
│   ├── basic_usage.py         # Basic pipeline usage
│   ├── custom_pipeline.py     # Custom pipeline configuration
│   └── custom_converter.py    # Extending with custom components
│
├── tests/                      # Unit tests
│   ├── __init__.py
│   ├── test_pipeline.py       # Pipeline tests
│   ├── test_data_loader.py    # Data loader tests
│   └── test_misp_converter.py # MISP converter tests
│
├── docs/                       # Documentation
│   ├── ARCHITECTURE.md        # Architecture and design patterns
│   └── USAGE.md              # Detailed usage guide
│
├── data/                       # Data directory (gitignored)
│   └── .gitkeep
│
├── results/                    # Results directory (gitignored)
│   └── .gitkeep
│
└── logs/                       # Logs directory (gitignored)
    └── .gitkeep
```

## Component Overview

### Core Architecture (src/pipeline/)

**Purpose**: Orchestrates the entire pipeline execution

**Key Components**:
- `Pipeline`: Main orchestrator that executes stages sequentially
- `PipelineStageInterface`: Base class for all stages
- `PipelineContext`: Data container passed between stages
- `PipelineBuilder`: Builds configured pipelines (Builder pattern)
- `PipelineObserver`: Observer interface for monitoring

**Design Patterns**: Pipeline, Observer, Builder

---

### Data Loading (src/data/)

**Purpose**: Load data from various sources

**Key Components**:
- `DataLoader`: Strategy interface for data loading
- `CSVDataLoader`: CSV file loader
- `MultiFileDataLoader`: Load multiple files
- `DataLoaderFactory`: Creates loader instances (Factory pattern)
- `DataLoaderStage`: Pipeline stage implementation

**Design Patterns**: Strategy, Factory

---

### MISP Conversion (src/misp/)

**Purpose**: Convert raw data to MISP objects

**Key Components**:
- `MISPObject`: MISP object representation
- `MISPRelationship`: Relationship between objects
- `MISPConverter`: Strategy interface for conversion
- `DefaultMISPConverter`: Baseline converter implementation
- `MISPConverterFactory`: Creates converter instances
- `MISPConverterStage`: Pipeline stage implementation

**Design Patterns**: Strategy, Factory

**Extension Point**: Create custom converters for domain-specific data

---

### Graph Generation (src/graph/)

**Purpose**: Build graph representations from MISP objects

**Key Components**:
- `GraphBuilder`: Builder interface for graphs
- `NetworkXGraphBuilder`: NetworkX graph builder
- `PyTorchGeometricGraphBuilder`: PyTorch Geometric builder
- `GraphBuilderFactory`: Creates builder instances
- `GraphGeneratorStage`: Pipeline stage implementation

**Design Patterns**: Builder, Factory

**Supports**: NetworkX graphs, PyTorch Geometric Data objects

---

### GNN Processing (src/models/)

**Purpose**: Train GNN models and generate embeddings

**Key Components**:
- `GNNModel`: Strategy interface for GNN models
- `BaselineGNNModel`: Baseline GNN implementation
- `Node2VecModel`: Node2Vec-based embeddings
- `GNNModelFactory`: Creates model instances
- `GNNProcessorStage`: Pipeline stage implementation

**Design Patterns**: Strategy, Factory

**Extension Point**: Add custom GNN architectures (GCN, GAT, GraphSAGE, etc.)

---

### Clustering (src/clustering/)

**Purpose**: Apply clustering algorithms to embeddings

**Key Components**:
- `ClusteringStrategy`: Strategy interface for clustering
- `KMeansClustering`: K-Means implementation
- `DBSCANClustering`: DBSCAN implementation
- `HierarchicalClustering`: Hierarchical clustering
- `ClusteringFactory`: Creates clustering instances
- `ClusteringProcessorStage`: Pipeline stage implementation

**Design Patterns**: Strategy, Factory

**Supports**: K-Means, DBSCAN, Hierarchical, extensible to others

---

### Storage (src/storage/)

**Purpose**: Persist results and artifacts

**Key Components**:
- `ResultRepository`: Repository interface for storage
- `FileSystemRepository`: File-based storage
- `ResultStorageStage`: Pipeline stage implementation

**Design Patterns**: Repository

**Extension Point**: Add database, cloud storage backends

---

### Visualization (src/visualization/)

**Purpose**: Generate visual representations

**Key Components**:
- `VisualizationGenerator`: Creates visualizations

**Capabilities**:
- Graph structure visualization
- Embedding projections (2D)
- Cluster distribution charts
- Metrics tables

---

### Utilities (src/utils/)

**Purpose**: Cross-cutting concerns

**Key Components**:
- `LoggerObserver`: Logs pipeline events
- `MetricsCollector`: Collects performance metrics

**Design Patterns**: Observer

---

## Data Flow

```
CSV Data → DataLoader → PipelineContext.raw_data
                              ↓
                      MISPConverter → PipelineContext.misp_objects
                              ↓
                      GraphBuilder → PipelineContext.graph
                              ↓
                        GNNModel → PipelineContext.embeddings
                              ↓
                    ClusteringAlgorithm → PipelineContext.clusters
                              ↓
                      ResultRepository → Files/Visualizations
```

## Design Principles

1. **Separation of Concerns**: Each component has a single responsibility
2. **Open/Closed Principle**: Open for extension, closed for modification
3. **Dependency Inversion**: Depend on abstractions, not concretions
4. **Interface Segregation**: Focused interfaces for specific purposes
5. **DRY (Don't Repeat Yourself)**: Reusable components and utilities

## Extensibility

The architecture is designed for easy extension:

1. **New Data Sources**: Implement `DataLoader` interface
2. **New Converters**: Implement `MISPConverter` interface
3. **New GNN Models**: Implement `GNNModel` interface
4. **New Clustering**: Implement `ClusteringStrategy` interface
5. **New Storage**: Implement `ResultRepository` interface
6. **New Stages**: Implement `PipelineStageInterface`

## Testing

- Unit tests for each component in isolation
- Integration tests for stage interactions
- Pipeline tests for end-to-end validation
- Mock objects for external dependencies

## Configuration

YAML-based configuration allows:
- Declarative pipeline setup
- Easy experimentation
- Reproducible results
- No code changes for parameter tuning
