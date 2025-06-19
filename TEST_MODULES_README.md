# Pattern Library Test Modules

This document describes the comprehensive test modules for the Pattern library, which automatically test algorithms across three different scales: **In-Memory**, **PySpark**, and **Coreset**.

## Overview

The Pattern library testing framework consists of three main test modules:

1. **`test_library_memory.py`** - In-memory scale testing
2. **`test_library_spark.py`** - Distributed PySpark scale testing  
3. **`test_library_coreset.py`** - Coreset-based efficient scale testing

Each module automatically discovers implemented algorithms, generates appropriate datasets, and evaluates performance using both default hyperparameters and Optuna optimization.

## Test Modules

### 1. In-Memory Scale Testing (`test_library_memory.py`)

**Purpose**: Tests algorithms on moderate-sized datasets that fit in memory.

**Features**:
- Automatic algorithm and metric discovery
- Benchmark dataset downloading (Iris, Wine, Karate Club, etc.)
- Synthetic data generation for all modalities
- Hyperparameter optimization with Optuna
- Comprehensive performance reporting

**Usage**:
```bash
python test_library_memory.py
```

**Datasets Tested**:
- **Attribute**: Iris, Wine, Breast Cancer, Seeds
- **Network**: Karate Club, Dolphins, Football, Political Books
- **Attributed Graph**: Cora, CiteSeer, PubMed

### 2. PySpark Scale Testing (`test_library_spark.py`)

**Purpose**: Tests algorithms on large-scale datasets using distributed processing.

**Features**:
- Distributed algorithm testing with PySpark
- Large-scale synthetic dataset generation
- Scalability analysis and performance metrics
- Spark session optimization
- Distributed result aggregation

**Requirements**:
```bash
pip install pyspark
```

**Usage**:
```bash
python test_library_spark.py
```

**Datasets Generated**:
- Large attribute datasets (50K-100K samples)
- Large network datasets (5K-10K nodes)
- High-dimensional scenarios

### 3. Coreset Scale Testing (`test_library_coreset.py`)

**Purpose**: Tests algorithms using coreset approximations for efficient large-scale processing.

**Features**:
- Coreset construction using multiple methods (k-means++, uniform sampling)
- Approximation quality analysis
- Efficiency and compression ratio metrics
- Scalable processing of large datasets
- Quality vs. efficiency trade-off analysis

**Usage**:
```bash
python test_library_coreset.py
```

**Coreset Methods**:
- K-means++ sampling
- Uniform random sampling
- Leverage score sampling (future)
- Density-based sampling (future)

## Data Modalities

All test modules support three data modalities:

### 1. Attribute Data (Features only)
- Traditional clustering datasets
- High-dimensional feature vectors
- Synthetic blob and mixture datasets

### 2. Network Data (Graph structure)
- Social networks
- Biological networks
- Synthetic networks (SBM, scale-free, small-world)

### 3. Attributed Graph Data (Features + Graph)
- Citation networks with paper features
- Social networks with user attributes
- Synthetic attributed graphs

## Configuration

### Algorithm Discovery
The test modules automatically discover algorithms from `MODEL_REGISTRY`:
- Filters algorithms by compatibility with each scale
- Infers modality (attribute, network, attributed_graph)
- Applies appropriate default parameters

### Hyperparameter Optimization
Uses multiple optimization strategies:
- **TPESearch**: Tree-structured Parzen Estimator
- **GridSearch**: Exhaustive grid search
- **RandomSearch**: Random parameter sampling

### Metrics
Evaluates using both standard and Pattern-specific metrics:
- **Standard**: ARI, NMI, Silhouette Score
- **Pattern Library**: Custom quality metrics from `METRIC_REGISTRY`

## Output and Results

### Result Files
Each test module generates:
- **Detailed CSV**: Complete test results with all metrics
- **Summary JSON**: Aggregated performance statistics
- **Log Files**: Detailed execution logs

### Result Structure
```
test_results_[scale]/
├── [scale]_detailed_results_YYYYMMDD_HHMMSS.csv
├── [scale]_summary_report_YYYYMMDD_HHMMSS.json
└── [scale]_test_log_YYYYMMDD_HHMMSS.log
```

### Key Metrics Reported
- **Success Rate**: Percentage of successful algorithm runs
- **Execution Time**: Average and per-algorithm timing
- **Quality Metrics**: Performance on benchmark datasets
- **Scalability Metrics**: Data size vs. performance analysis
- **Approximation Quality** (Coreset): Quality of coreset approximations

## Running All Tests

To run comprehensive testing across all scales:

```bash
# Run in sequence
python test_library_memory.py
python test_library_spark.py    # Requires PySpark
python test_library_coreset.py

# Or create a master script
python -c "
import subprocess
import sys

tests = ['test_library_memory.py', 'test_library_coreset.py']
try:
    import pyspark
    tests.append('test_library_spark.py')
except ImportError:
    print('Skipping Spark tests - PySpark not available')

for test in tests:
    print(f'Running {test}...')
    subprocess.run([sys.executable, test])
"
```

## Dependencies

### Core Dependencies (all modules):
```
numpy
pandas
scikit-learn
networkx
optuna
requests
```

### PySpark Module Additional:
```
pyspark
```

### Pattern Library:
```
# Your Pattern library components
config.registries
config.validator
core.factory
core.logger
data.loaders
optimization.strategies
```

## Customization

### Adding New Datasets
1. **Memory**: Extend `BenchmarkDataManager.benchmark_datasets`
2. **Spark**: Extend `SparkDataManager.dataset_configs`
3. **Coreset**: Extend `CoresetDataManager.coreset_configs`

### Adding New Algorithms
Algorithms are automatically discovered from `MODEL_REGISTRY`. Ensure your algorithms:
- Are registered in the registry
- Have proper parameter documentation
- Support the expected data loader interface

### Adding New Metrics
Metrics are automatically discovered from `METRIC_REGISTRY`. Custom metrics should:
- Implement the metric interface
- Handle different data modalities appropriately
- Return numeric scores (not NaN)

## Performance Expectations

### Memory Scale
- **Dataset Size**: 100-10,000 samples
- **Execution Time**: 1-60 seconds per test
- **Memory Usage**: < 1GB

### Spark Scale  
- **Dataset Size**: 10,000-100,000 samples
- **Execution Time**: 10-300 seconds per test
- **Memory Usage**: Distributed across cluster

### Coreset Scale
- **Original Size**: 10,000-50,000 samples
- **Coreset Size**: 500-5,000 samples
- **Compression Ratio**: 5x-100x
- **Execution Time**: 5-120 seconds per test

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure Pattern library is in Python path
2. **PySpark Issues**: Check Java installation and SPARK_HOME
3. **Memory Errors**: Reduce dataset sizes in configurations
4. **Algorithm Failures**: Check algorithm parameter compatibility
5. **Network Download Failures**: Check internet connection and URLs

### Debug Mode
Enable detailed logging by modifying the logging level:
```python
logger.setLevel(logging.DEBUG)
```

### Selective Testing
Run specific algorithms by modifying the discovery methods:
```python
# In any test module
def discover_algorithms(self):
    # Filter to specific algorithms
    target_algorithms = ['kmeans', 'dbscan']
    # ... filter logic
```

## Future Enhancements

### Planned Features
- GPU-accelerated testing module
- Distributed coreset construction
- Real-time performance monitoring
- Automated benchmark comparison
- CI/CD integration
- Interactive result visualization

### Contributing
To extend the testing framework:
1. Follow existing module structure
2. Implement proper error handling
3. Add comprehensive logging
4. Update this documentation
5. Test with multiple algorithm types

## License

This testing framework follows the same license as the Pattern library. 