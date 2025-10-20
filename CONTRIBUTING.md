# Contributing Guide

## How to Contribute

We welcome contributions to improve the GNN Campaign Detection pipeline! Here are ways you can contribute:

### 1. Report Bugs

If you find a bug, please create an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (Python version, OS, etc.)

### 2. Suggest Enhancements

For feature requests:
- Describe the enhancement
- Explain the use case
- Provide examples if possible

### 3. Submit Pull Requests

#### Adding a New Component

**Example: Adding a new clustering algorithm**

1. Create your implementation:

```python
# src/clustering/processor.py

class MyNewClustering(ClusteringStrategy):
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2

    def fit_predict(self, embeddings):
        # Your implementation
        return cluster_labels

    def get_metrics(self):
        return {'metric1': value1}
```

2. Register in factory:

```python
# src/clustering/processor.py

class ClusteringFactory:
    @staticmethod
    def create(algorithm: str, **kwargs):
        # ... existing code ...
        elif algorithm == 'mynew':
            return MyNewClustering(**kwargs)
```

3. Add tests:

```python
# tests/test_clustering.py

def test_my_new_clustering():
    clustering = MyNewClustering(param1=1, param2=2)
    embeddings = np.random.randn(100, 64)
    labels = clustering.fit_predict(embeddings)
    assert len(labels) == 100
```

4. Update documentation

5. Submit PR with clear description

## Code Style

- Follow PEP 8
- Use type hints
- Add docstrings to all public methods
- Keep functions focused and small

## Testing

Run tests before submitting:

```bash
pytest tests/ -v
```

## Documentation

- Update relevant documentation files
- Add examples if introducing new features
- Keep README.md up to date

## Questions?

Open an issue for any questions or discussions!
