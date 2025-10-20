# GNN Campaign Detection Pipeline

A scalable and maintainable research pipeline for detecting campaigns using Graph Neural Networks.

## Architecture Overview

This project implements a modular pipeline architecture with the following stages:

1. **Data Loading** - Load CSV data files
2. **MISP Conversion** - Convert data to MISP objects
3. **Graph Generation** - Generate graph representations
4. **GNN Processing** - Generate embeddings using Graph Neural Networks
5. **Clustering** - Apply clustering algorithms
6. **Results Storage** - Store numerical results and visualizations

## Design Patterns Used

- **Pipeline Pattern**: Sequential processing stages with clear interfaces
- **Strategy Pattern**: Interchangeable algorithms (GNN models, clustering methods)
- **Factory Pattern**: Creating different types of objects (MISP objects, graphs)
- **Repository Pattern**: Data access and storage abstraction
- **Observer Pattern**: Logging and monitoring throughout the pipeline
- **Builder Pattern**: Complex object construction (graphs, configurations)

## Project Structure

```
GNN-Campaign-Detection/
├── src/
│   ├── pipeline/           # Core pipeline components
│   ├── data/               # Data loading and processing
│   ├── misp/               # MISP object conversion
│   ├── graph/              # Graph generation and management
│   ├── models/             # GNN model implementations
│   ├── clustering/         # Clustering algorithms
│   ├── storage/            # Results storage
│   ├── visualization/      # Visualization components
│   └── utils/              # Utility functions
├── config/                 # Configuration files
├── tests/                  # Unit and integration tests
├── examples/               # Example usage scripts
├── data/                   # Data directory (gitignored)
├── results/                # Results directory (gitignored)
└── requirements.txt        # Python dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from src.pipeline import PipelineBuilder
from src.config import PipelineConfig

# Configure pipeline
config = PipelineConfig.from_file('config/default_config.yaml')

# Build and run pipeline
pipeline = PipelineBuilder(config).build()
results = pipeline.execute()

# Access results
print(results.metrics)
results.save()
```

## Extending the Pipeline

### Adding a New GNN Model

```python
from src.models.base import GNNModel

class CustomGNN(GNNModel):
    def train(self, graph):
        # Implementation
        pass
    
    def generate_embeddings(self, graph):
        # Implementation
        pass
```

### Adding a New Clustering Algorithm

```python
from src.clustering.base import ClusteringStrategy

class CustomClustering(ClusteringStrategy):
    def fit_predict(self, embeddings):
        # Implementation
        pass
```
