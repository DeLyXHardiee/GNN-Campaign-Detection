# Architecture Diagrams

## Pipeline Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        PIPELINE                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │              PipelineContext (Shared State)             │    │
│  └────────────────────────────────────────────────────────┘    │
│                              │                                   │
│  ┌───────────────────────────▼──────────────────────────────┐  │
│  │  Stage 1: Data Loading                                    │  │
│  │  ┌──────────┐  Uses   ┌──────────────┐                  │  │
│  │  │ Factory  │───────>│ DataLoader    │                  │  │
│  │  └──────────┘         │ (Strategy)    │                  │  │
│  │                       └──────────────┘                  │  │
│  │  Output: context.raw_data                                │  │
│  └───────────────────────────┬──────────────────────────────┘  │
│                              │                                   │
│  ┌───────────────────────────▼──────────────────────────────┐  │
│  │  Stage 2: MISP Conversion                                 │  │
│  │  ┌──────────┐  Uses   ┌──────────────┐                  │  │
│  │  │ Factory  │───────>│ MISPConverter │                  │  │
│  │  └──────────┘         │ (Strategy)    │                  │  │
│  │                       └──────────────┘                  │  │
│  │  Output: context.misp_objects                            │  │
│  └───────────────────────────┬──────────────────────────────┘  │
│                              │                                   │
│  ┌───────────────────────────▼──────────────────────────────┐  │
│  │  Stage 3: Graph Generation                                │  │
│  │  ┌──────────┐  Uses   ┌──────────────┐                  │  │
│  │  │ Factory  │───────>│ GraphBuilder  │                  │  │
│  │  └──────────┘         │ (Builder)     │                  │  │
│  │                       └──────────────┘                  │  │
│  │  Output: context.graph                                   │  │
│  └───────────────────────────┬──────────────────────────────┘  │
│                              │                                   │
│  ┌───────────────────────────▼──────────────────────────────┐  │
│  │  Stage 4: GNN Processing                                  │  │
│  │  ┌──────────┐  Uses   ┌──────────────┐                  │  │
│  │  │ Factory  │───────>│ GNNModel      │                  │  │
│  │  └──────────┘         │ (Strategy)    │                  │  │
│  │                       └──────────────┘                  │  │
│  │  Output: context.embeddings                              │  │
│  └───────────────────────────┬──────────────────────────────┘  │
│                              │                                   │
│  ┌───────────────────────────▼──────────────────────────────┐  │
│  │  Stage 5: Clustering                                      │  │
│  │  ┌──────────┐  Uses   ┌──────────────────┐              │  │
│  │  │ Factory  │───────>│ ClusteringStrategy│              │  │
│  │  └──────────┘         │ (Strategy)        │              │  │
│  │                       └──────────────────┘              │  │
│  │  Output: context.clusters                                │  │
│  └───────────────────────────┬──────────────────────────────┘  │
│                              │                                   │
│  ┌───────────────────────────▼──────────────────────────────┐  │
│  │  Stage 6: Result Storage                                  │  │
│  │  Uses   ┌──────────────────┐                             │  │
│  │  ──────>│ ResultRepository  │                             │  │
│  │         │ (Repository)      │                             │  │
│  │         └──────────────────┘                             │  │
│  │  Output: context.results (file paths)                    │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Observers watching all stages:                                 │
│  ┌──────────────┐  ┌──────────────────┐                       │
│  │LoggerObserver│  │MetricsCollector  │  ...                  │
│  └──────────────┘  └──────────────────┘                       │
└─────────────────────────────────────────────────────────────────┘
```

## Class Hierarchy - Pipeline Components

```
PipelineStageInterface (ABC)
│
├── DataLoaderStage
├── MISPConverterStage
├── GraphGeneratorStage
├── GNNProcessorStage
├── ClusteringProcessorStage
└── ResultStorageStage
```

## Class Hierarchy - Strategy Implementations

```
DataLoader (ABC)
├── CSVDataLoader
└── MultiFileDataLoader

MISPConverter (ABC)
└── DefaultMISPConverter
    └── [Your Custom Converter]

GraphBuilder (ABC)
├── NetworkXGraphBuilder
└── PyTorchGeometricGraphBuilder

GNNModel (ABC)
├── BaselineGNNModel
└── Node2VecModel
    └── [Your Custom GNN]

ClusteringStrategy (ABC)
├── KMeansClustering
├── DBSCANClustering
└── HierarchicalClustering

ResultRepository (ABC)
└── FileSystemRepository
    └── [DatabaseRepository, S3Repository, ...]
```

## Observer Pattern

```
┌─────────────────┐
│    Pipeline     │
│                 │
│  ┌───────────┐  │      Notifies
│  │  Stage 1  │──┼───────────────┐
│  └───────────┘  │               │
│  ┌───────────┐  │               │
│  │  Stage 2  │──┼───────────┐   │
│  └───────────┘  │           │   │
│  ┌───────────┐  │           │   │
│  │  Stage 3  │──┼────┐      │   │
│  └───────────┘  │    │      │   │
└─────────────────┘    │      │   │
                       │      │   │
                       ▼      ▼   ▼
            ┌─────────────────────────────┐
            │   PipelineObserver (ABC)    │
            └─────────────────────────────┘
                       │      │
            ┌──────────┴──┐   └──────────┐
            ▼             ▼              ▼
    ┌──────────────┐ ┌──────────┐  ┌─────────┐
    │LoggerObserver│ │  Metrics │  │ Custom  │
    │              │ │Collector │  │Observer │
    └──────────────┘ └──────────┘  └─────────┘
```

## Factory Pattern Structure

```
┌──────────────────────┐
│   Configuration      │
│   (YAML/Dict)        │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│     Factory          │
│   (Static Method)    │
└──────────┬───────────┘
           │ Creates based on type
           ▼
┌──────────────────────┐
│   Concrete           │
│   Implementation     │
│   (Strategy)         │
└──────────────────────┘

Example:
  config: {type: "kmeans", n_clusters: 5}
     │
     ▼
  ClusteringFactory.create("kmeans", n_clusters=5)
     │
     ▼
  KMeansClustering(n_clusters=5)
```

## Builder Pattern - Pipeline Construction

```
PipelineConfig
     │
     ▼
PipelineBuilder
     │
     ├── add_observer(logger)
     ├── add_observer(metrics)
     │
     ├── Create Stage 1 (from config)
     ├── Create Stage 2 (from config)
     ├── Create Stage 3 (from config)
     ├── Create Stage 4 (from config)
     ├── Create Stage 5 (from config)
     └── Create Stage 6 (from config)
     │
     ▼
build() → Pipeline
           └── Ready to execute()
```

## Data Flow with Context

```
┌─────────────────────────────────────────────┐
│         PipelineContext                      │
├─────────────────────────────────────────────┤
│ raw_data:        None → DataFrame            │
│ misp_objects:    None → List[MISPObject]    │
│ graph:           None → NetworkX.Graph      │
│ embeddings:      None → np.ndarray          │
│ clusters:        None → np.ndarray          │
│ results:         None → Dict[paths]         │
│ metadata:        {} → {...}                 │
└─────────────────────────────────────────────┘
         Passed through stages
              │
              ├─> Stage 1: Updates raw_data
              ├─> Stage 2: Reads raw_data, Updates misp_objects
              ├─> Stage 3: Reads misp_objects, Updates graph
              ├─> Stage 4: Reads graph, Updates embeddings
              ├─> Stage 5: Reads embeddings, Updates clusters
              └─> Stage 6: Reads all, Updates results
```

## Component Interaction Example

```
User Code:
  │
  ├─> Load PipelineConfig (from YAML)
  │
  ├─> Create PipelineBuilder(config)
  │
  ├─> builder.add_observer(logger)
  │
  ├─> pipeline = builder.build()
  │      │
  │      └─> For each stage config:
  │            Factory.create(stage_type, **config)
  │
  └─> context = pipeline.execute()
         │
         └─> For each stage:
               │
               ├─> stage.validate(context)
               │
               ├─> stage.execute(context)
               │     │
               │     └─> stage.notify_observers(event, data)
               │               │
               │               └─> observer.update(event, data)
               │
               └─> Update context with results
```

## Extension Points

```
Want to add...                    Implement...              Register in...

New Data Source              →    DataLoader              →  DataLoaderFactory
New MISP Converter           →    MISPConverter           →  MISPConverterFactory
New Graph Type               →    GraphBuilder            →  GraphBuilderFactory
New GNN Architecture         →    GNNModel                →  GNNModelFactory
New Clustering Algorithm     →    ClusteringStrategy      →  ClusteringFactory
New Storage Backend          →    ResultRepository        →  (use directly)
New Pipeline Stage           →    PipelineStageInterface  →  PipelineBuilder
New Observer/Monitor         →    PipelineObserver        →  (add to pipeline)
```

## Design Patterns Summary

| Pattern      | Purpose                          | Location                    |
|--------------|----------------------------------|-----------------------------|
| Pipeline     | Sequential stage execution       | src/pipeline/base.py        |
| Strategy     | Interchangeable algorithms       | All stage implementations   |
| Factory      | Object creation                  | All *Factory classes        |
| Observer     | Event notification              | src/utils/logging.py        |
| Builder      | Complex object construction      | src/pipeline/builder.py     |
| Repository   | Data persistence abstraction     | src/storage/saver.py        |
