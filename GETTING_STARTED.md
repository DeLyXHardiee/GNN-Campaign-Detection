# Getting Started Guide

Welcome! This guide will help you get up and running with the GNN Campaign Detection Pipeline.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git (optional, for cloning)

## Installation

### Step 1: Set Up Environment

```powershell
# Navigate to project directory
cd .../GNN-Campaign-Detection

# (Optional) Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Verify Installation

```powershell
# Check Python version
python --version

# Verify key packages
python -c "import pandas; import numpy; import networkx; print('OK')"
```

## Quick Start (5 Minutes)

### Option 1: Run the Demo

The quickest way to see the pipeline in action:

```powershell
python quickstart.py
```

This will:
1. ✅ Create sample data
2. ✅ Generate configuration
3. ✅ Run the complete pipeline
4. ✅ Save results to `results/` directory
5. ✅ Generate visualizations

**Expected Output:**
```
============================================================
GNN CAMPAIGN DETECTION - QUICK START DEMO
============================================================

Creating sample data...
Sample data created: data\sample_data.csv
  Rows: 100
  Columns: ['type', 'value', 'category', 'timestamp', 'severity']

Updating configuration...
Configuration updated: config\demo_config.yaml

============================================================
RUNNING PIPELINE DEMO
============================================================

[YYYY-MM-DD HH:MM:SS] Pipeline execution started
[YYYY-MM-DD HH:MM:SS] Stage started: DataLoaderStage
...
============================================================
PIPELINE COMPLETED SUCCESSFULLY
============================================================

Results Summary:
  Results directory: {...}

✅ Demo completed successfully!
```

### Option 2: Use Example Scripts

```powershell
# Basic usage
python examples\basic_usage.py

# Custom pipeline
python examples\custom_pipeline.py
```

## Understanding the Results

After running the pipeline, check the `results/` directory:

```
results/
└── run_YYYYMMDD_HHMMSS/
    ├── metrics.json              # Numerical metrics
    ├── embeddings.npy            # Node embeddings
    ├── clusters.npy              # Cluster assignments
    └── visualizations/
        ├── graph.png             # Graph structure
        ├── embeddings.png        # 2D embedding projection
        └── cluster_distribution.png  # Cluster sizes
```

### View Metrics

```powershell
# View metrics
python -c "import json; print(json.dumps(json.load(open('results/run_*/metrics.json')), indent=2))"
```

### Load Results in Python

```python
import numpy as np
import json

# Load embeddings
embeddings = np.load('results/run_YYYYMMDD_HHMMSS/embeddings.npy')
print(f"Embedding shape: {embeddings.shape}")

# Load clusters
clusters = np.load('results/run_YYYYMMDD_HHMMSS/clusters.npy')
print(f"Unique clusters: {len(set(clusters))}")

# Load metrics
with open('results/run_YYYYMMDD_HHMMSS/metrics.json') as f:
    metrics = json.load(f)
print(f"Pipeline metrics: {metrics}")
```

## Working with Your Own Data

### Step 1: Prepare Your Data

Create a CSV file with your data:

```csv
type,value,category,feature1,feature2
event,val1,cat1,0.5,0.8
indicator,val2,cat2,0.3,0.6
attribute,val3,cat1,0.7,0.4
```

Save it as `data/my_data.csv`

### Step 2: Update Configuration

Edit `config/default_config.yaml`:

```yaml
data_loading:
  loader_type: "csv"
  data_source: "data/my_data.csv"  # Your file path

misp_conversion:
  converter_kwargs:
    object_type_column: "type"  # Column with entity types
```

### Step 3: Run Pipeline

```python
from src.pipeline import PipelineBuilder, PipelineConfig

config = PipelineConfig.from_file('config/default_config.yaml')
pipeline = PipelineBuilder(config).build()
context = pipeline.execute()

print(f"Results: {context.results}")
```

## Experimenting with Algorithms

### Try Different Clustering Algorithms

**K-Means:**
```yaml
clustering:
  algorithm: "kmeans"
  algorithm_kwargs:
    n_clusters: 5
```

**DBSCAN:**
```yaml
clustering:
  algorithm: "dbscan"
  algorithm_kwargs:
    eps: 0.5
    min_samples: 5
```

**Hierarchical:**
```yaml
clustering:
  algorithm: "hierarchical"
  algorithm_kwargs:
    n_clusters: 5
    linkage: "ward"
```

### Try Different GNN Models

**Baseline:**
```yaml
gnn_processing:
  model_type: "baseline"
  model_kwargs:
    embedding_dim: 64
    num_layers: 2
```

**Node2Vec:**
```yaml
gnn_processing:
  model_type: "node2vec"
  model_kwargs:
    embedding_dim: 128
    walk_length: 10
    num_walks: 80
```

## Common Tasks

### Task 1: Compare Multiple Configurations

```python
from src.pipeline import PipelineBuilder, PipelineConfig

configs = [
    'config/config_kmeans.yaml',
    'config/config_dbscan.yaml',
    'config/config_hierarchical.yaml'
]

results = {}
for config_path in configs:
    config = PipelineConfig.from_file(config_path)
    pipeline = PipelineBuilder(config).build()
    context = pipeline.execute()
    
    results[config_path] = {
        'clusters': len(set(context.clusters)),
        'paths': context.results
    }

print(results)
```

### Task 2: Process Multiple Files

```python
import glob
from pathlib import Path

data_files = glob.glob('data/*.csv')

for data_file in data_files:
    print(f"Processing {data_file}...")
    
    config = PipelineConfig.from_dict({
        'data_loading': {
            'loader_type': 'csv',
            'data_source': data_file
        },
        # ... other config sections
    })
    
    pipeline = PipelineBuilder(config).build()
    context = pipeline.execute()
    print(f"✅ Completed: {context.results}")
```

### Task 3: Custom Monitoring

```python
from src.pipeline import PipelineBuilder, PipelineConfig
from src.utils.logging import LoggerObserver, MetricsCollector

# Create observers
logger = LoggerObserver(log_file='logs/my_run.log', verbose=True)
metrics = MetricsCollector()

# Build pipeline with observers
config = PipelineConfig.from_file('config/default_config.yaml')
pipeline = (
    PipelineBuilder(config)
    .add_observer(logger)
    .add_observer(metrics)
    .build()
)

# Execute
context = pipeline.execute()

# Get metrics
collected = metrics.get_metrics()
print(f"Execution time: {collected.get('total_duration_seconds')} seconds")
print(f"Clusters found: {collected.get('n_clusters')}")
```

## Extending the Pipeline

### Add a Custom MISP Converter

```python
from src.misp.converter import MISPConverter, MISPObject
import pandas as pd

class MyCustomConverter(MISPConverter):
    def convert(self, data: pd.DataFrame):
        objects = []
        for _, row in data.iterrows():
            obj = MISPObject(
                object_type=row['my_type_column'],
                attributes={
                    'id': row['id'],
                    'feature1': row['feature1']
                }
            )
            objects.append(obj)
        return objects
    
    def extract_relationships(self, objects):
        # Your relationship logic
        return []

# Use it
from src.misp.converter import MISPConverterStage

stage = MISPConverterStage({'converter_type': 'default'})
stage.converter = MyCustomConverter()
```

### Add a Custom Observer

```python
from src.pipeline.base import PipelineObserver

class MyCustomObserver(PipelineObserver):
    def update(self, event: str, data: dict):
        if event == "clustering_completed":
            clusters = data.get('unique_clusters', 0)
            print(f"🎯 Found {clusters} clusters!")

# Use it
pipeline.add_observer(MyCustomObserver())
```

## Troubleshooting

### Issue: ModuleNotFoundError

**Solution:**
```powershell
# Make sure you're in the project directory
cd ...\GNN-Campaign-Detection

# Reinstall dependencies
pip install -r requirements.txt

# Add project to Python path (temporary)
$env:PYTHONPATH = "...\GNN-Campaign-Detection"
```

### Issue: FileNotFoundError for data

**Solution:**
```python
# Check if file exists
from pathlib import Path
data_path = Path('data/my_data.csv')
print(f"File exists: {data_path.exists()}")

# Use absolute path
config['data_loading']['data_source'] = str(data_path.absolute())
```

### Issue: Empty visualizations

**Solution:**
```powershell
# Install matplotlib
pip install matplotlib

# Check if results directory exists
ls results/
```

## Next Steps

### 1. Read the Documentation
- 📖 [ARCHITECTURE.md](ARCHITECTURE.md) - Understand the design
- 📖 [USAGE.md](USAGE.md) - Detailed usage guide
- 📖 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - Code organization
- 📖 [DIAGRAMS.md](DIAGRAMS.md) - Visual architecture

### 2. Explore Examples
- 💡 `examples/basic_usage.py` - Simple pipeline usage
- 💡 `examples/custom_pipeline.py` - Custom configuration
- 💡 `examples/custom_converter.py` - Extension example

### 3. Implement Your Research
- 🔬 Add your domain-specific MISP converter
- 🔬 Implement your GNN model (GCN, GAT, etc.)
- 🔬 Add your custom clustering algorithms
- 🔬 Extend with your features

### 4. Run Experiments
- 🧪 Try different configurations
- 🧪 Compare algorithms
- 🧪 Analyze results
- 🧪 Generate visualizations

## Getting Help

### Resources
- 📚 Check documentation in `docs/` folder
- 💻 Review example scripts in `examples/` folder
- 🧪 Run tests: `pytest tests/ -v`
- 📝 Check logs in `logs/` folder

### Common Questions

**Q: How do I add my own GNN model?**
A: Implement the `GNNModel` interface and register in `GNNModelFactory`. See `examples/custom_converter.py` for pattern.

**Q: Can I use my own data format?**
A: Yes! Implement a custom `DataLoader` or preprocess to CSV format.

**Q: How do I save results to a database?**
A: Implement `ResultRepository` interface with your database logic.

**Q: Can I run stages individually?**
A: Yes! Each stage is independent. See `examples/custom_pipeline.py`.

## Success Checklist

After following this guide, you should be able to:

- ✅ Run the demo pipeline
- ✅ Process your own CSV data
- ✅ Modify configuration files
- ✅ View and analyze results
- ✅ Experiment with different algorithms
- ✅ Understand the architecture
- ✅ Extend with custom components

## You're Ready!

You now have a complete, production-ready pipeline for your GNN research. The architecture is designed to grow with your needs - start simple, add complexity as required.

**Happy researching! 🚀**
