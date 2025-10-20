# Usage Guide

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd GNN-Campaign-Detection

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Your Data

Place your CSV data file in the `data/` directory:

```
data/
└── input.csv
```

Your CSV should contain the data you want to analyze. The structure depends on your use case.

### 3. Configure the Pipeline

Edit `config/default_config.yaml` to match your data structure:

```yaml
data_loading:
  data_source: "data/input.csv"

misp_conversion:
  converter_kwargs:
    object_type_column: "type"  # Column containing entity types
```

### 4. Run the Pipeline

```bash
python examples/basic_usage.py
```

### 5. View Results

Results are saved to the `results/` directory:

```
results/
└── run_YYYYMMDD_HHMMSS/
    ├── metrics.json
    ├── embeddings.npy
    ├── clusters.npy
    └── visualizations/
        ├── graph.png
        ├── embeddings.png
        └── cluster_distribution.png
```

## Detailed Usage

### Configuration Options

#### Data Loading

```yaml
data_loading:
  loader_type: "csv"  # or "multi_file"
  data_source: "data/input.csv"
  loader_kwargs:
    sep: ","
    encoding: "utf-8"
```

#### MISP Conversion

```yaml
misp_conversion:
  converter_type: "default"
  converter_kwargs:
    object_type_column: "type"
    attribute_columns: null  # null = use all columns
```

#### Graph Generation

```yaml
graph_generation:
  builder_type: "networkx"  # or "pytorch_geometric"
  builder_kwargs:
    directed: true
    include_node_attributes: true
    include_edge_attributes: true
```

#### GNN Processing

```yaml
gnn_processing:
  model_type: "baseline"  # or "node2vec"
  model_kwargs:
    embedding_dim: 64
    num_layers: 2
  training:
    epochs: 100
```

#### Clustering

```yaml
clustering:
  algorithm: "kmeans"  # or "dbscan", "hierarchical"
  algorithm_kwargs:
    n_clusters: 5
    random_state: 42
```

### Programmatic Usage

#### Basic Usage

```python
from src.pipeline import PipelineBuilder, PipelineConfig

# Load configuration
config = PipelineConfig.from_file('config/default_config.yaml')

# Build and execute pipeline
pipeline = PipelineBuilder(config).build()
context = pipeline.execute()

# Access results
print(f"Embeddings shape: {context.embeddings.shape}")
print(f"Number of clusters: {len(set(context.clusters))}")
```

#### Custom Configuration

```python
from src.pipeline import PipelineConfig

# Create configuration from dictionary
config_dict = {
    'data_loading': {
        'loader_type': 'csv',
        'data_source': 'data/my_data.csv'
    },
    'gnn_processing': {
        'model_type': 'baseline',
        'model_kwargs': {'embedding_dim': 128}
    },
    # ... other stages
}

config = PipelineConfig.from_dict(config_dict)
```

#### Adding Observers

```python
from src.pipeline import PipelineBuilder, PipelineConfig
from src.utils.logging import LoggerObserver, MetricsCollector

config = PipelineConfig.from_file('config/default_config.yaml')

# Create custom observers
logger = LoggerObserver(log_file='logs/custom.log')
metrics = MetricsCollector()

# Build pipeline with observers
pipeline = (
    PipelineBuilder(config)
    .add_observer(logger)
    .add_observer(metrics)
    .build()
)

context = pipeline.execute()

# Get collected metrics
print(metrics.get_metrics())
```

### Custom Components

#### Custom MISP Converter

```python
from src.misp.converter import MISPConverter, MISPObject

class CustomConverter(MISPConverter):
    def convert(self, data):
        # Your conversion logic
        objects = []
        for _, row in data.iterrows():
            obj = MISPObject(
                object_type=row['type'],
                attributes=row.to_dict()
            )
            objects.append(obj)
        return objects

    def extract_relationships(self, objects):
        # Your relationship logic
        return []

# Use in pipeline
from src.misp.converter import MISPConverterStage

stage = MISPConverterStage({'converter_type': 'default'})
stage.converter = CustomConverter()
```

#### Custom GNN Model

```python
from src.models.processor import GNNModel
import numpy as np

class CustomGNN(GNNModel):
    def __init__(self, embedding_dim=64):
        self.embedding_dim = embedding_dim

    def train(self, graph, **kwargs):
        # Your training logic
        pass

    def generate_embeddings(self, graph):
        # Your embedding generation
        num_nodes = graph.number_of_nodes()
        return np.random.randn(num_nodes, self.embedding_dim)

    def save_model(self, path):
        # Save logic
        pass

    def load_model(self, path):
        # Load logic
        pass
```

## Common Workflows

### Experimenting with Different Algorithms

Create multiple configuration files:

```bash
config/
├── kmeans_config.yaml
├── dbscan_config.yaml
└── hierarchical_config.yaml
```

Run experiments:

```python
configs = [
    'config/kmeans_config.yaml',
    'config/dbscan_config.yaml',
    'config/hierarchical_config.yaml'
]

for config_path in configs:
    config = PipelineConfig.from_file(config_path)
    pipeline = PipelineBuilder(config).build()
    context = pipeline.execute()
    print(f"Results for {config_path}: {context.results}")
```

### Batch Processing Multiple Files

```python
import glob
from pathlib import Path

data_files = glob.glob('data/*.csv')

for data_file in data_files:
    # Update config with current file
    config_dict = {
        'data_loading': {
            'loader_type': 'csv',
            'data_source': data_file
        },
        # ... rest of config
    }

    config = PipelineConfig.from_dict(config_dict)
    pipeline = PipelineBuilder(config).build()

    try:
        context = pipeline.execute()
        print(f"Processed {data_file} successfully")
    except Exception as e:
        print(f"Failed to process {data_file}: {e}")
```

### Loading and Analyzing Results

```python
import numpy as np
import json

# Load saved results
embeddings = np.load('results/run_20231020_120000/embeddings.npy')
clusters = np.load('results/run_20231020_120000/clusters.npy')

with open('results/run_20231020_120000/metrics.json') as f:
    metrics = json.load(f)

# Analyze
print(f"Embedding dimensions: {embeddings.shape}")
print(f"Number of clusters: {len(set(clusters))}")
print(f"Metrics: {metrics}")
```

## Troubleshooting

### Common Issues

**Issue**: `ValueError: data_source must be specified in config`

**Solution**: Ensure your configuration file has a `data_source` field under `data_loading`.

**Issue**: `Validation failed for stage`

**Solution**: Check that previous stages completed successfully and required data is present.

**Issue**: Import errors

**Solution**: Make sure all dependencies are installed: `pip install -r requirements.txt`

### Debugging

Enable verbose logging:

```python
from src.utils.logging import LoggerObserver

logger = LoggerObserver(verbose=True)
pipeline.add_observer(logger)
```

Check log files in `logs/` directory for detailed execution traces.

## Performance Tips

1. **Use appropriate data types**: Ensure your CSV uses efficient data types
2. **Adjust batch sizes**: For large graphs, consider processing in batches
3. **Cache intermediate results**: Save graph and embeddings for reuse
4. **Use GPU**: If available, modify GNN implementations to use GPU acceleration

## Next Steps

- Read [ARCHITECTURE.md](ARCHITECTURE.md) to understand the design
- See [examples/](../examples/) for more usage patterns
- Extend the pipeline with your own custom components
