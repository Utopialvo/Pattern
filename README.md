# Pattern

**Scalable Unsupervised Learning Library for Multiple Data Types**

## Description

Pattern is a comprehensive unsupervised learning library designed to handle diverse data types and processing modes:

### **Supported Data Types**
- **🔢 Attributes/Features**: Traditional tabular data for feature-based clustering
- **🕸️ Graph/Networks**: Pure network data for graph-based clustering algorithms
- **🔗 Attributed Networks**: Combined feature and graph data for advanced clustering

### **Processing Modes**
- **🐼 Pandas**: Single-machine processing for smaller datasets
- **⚡ Apache Spark**: Distributed processing for large-scale data

### **Key Features**
- **Multi-Modal Data Support**: Seamlessly handle tabular, graph, and attributed network data
- **Dual Processing Backends**: Choose between pandas and Spark based on your data scale
- **Extensible Architecture**: Plugin-based system for algorithms, metrics, and preprocessing
- **Hyperparameter Optimization**: Advanced optimization with Optuna (TPE, Grid, Random)
- **Comprehensive Metrics**: Evaluation metrics tailored for different data types
- **Rich Visualization**: Data-type-aware visualization and statistical analysis
- **Production Ready**: Robust error handling, logging, and resource management

## Algorithms

### **Attribute-Based Clustering**
- **KMeans**: Traditional centroid-based clustering
- **DBSCAN**: Density-based clustering with noise detection

### **Graph-Based Clustering** 
- **Louvain**: Community detection via modularity optimization
- **Spectral**: Spectral graph clustering using eigendecomposition

### **Attributed Graph Clustering**
- **DMoN (Deep Modularity Networks)**: Deep learning approach for attributed graphs

## Metrics

### **Attribute Metrics**
- **Silhouette Score**: Cluster cohesion and separation
- **Calinski-Harabasz**: Variance ratio criterion
- **Davies-Bouldin**: Average similarity measure

### **Graph Metrics**
- **Modularity**: Community structure quality
- **Density Modularity**: Weighted community evaluation

### **Network-Specific Metrics**
- **ANUI**: Attributed Network Unsupervised Index
- **AVU/AVI**: Attributed Validation metrics

## Requirements

- **Python**: 3.7+ (recommended: 3.9+)
- **Apache Spark**: 3.3.1+ (optional, for distributed processing)

### Core Dependencies
```
joblib>=1.4.2
matplotlib>=3.10.3
networkx>=3.4.1
numpy>=2.2.6
optuna>=4.3.0
pandas>=2.0.3
pyspark>=3.3.1
scikit-learn>=1.6.1
scipy>=1.15.3
seaborn>=0.13.2
statsmodels>=0.14.4
torch>=2.7.0
torch-geometric>=2.6.1
tqdm>=4.66.5
```

## Installation

```bash
git clone https://github.com/Utopialvo/Pattern.git
cd Pattern
pip install -r requirements.txt
```

## Quick Start

### 1. Attribute-Based Clustering
```bash
# Single-machine tabular data clustering
python main.py config_attributes.json
```

### 2. Graph Clustering
```bash
# Network/graph-only clustering
python main.py config_graph.json
```

### 3. Attributed Graph Clustering
```bash
# Combined feature + graph clustering with Spark
python main.py config_attributed_graph.json
```

## Configuration Examples

### Attributes/Features Configuration
```json
{
  "data_source": "pandas",
  "data_type": "attributes",
  "features": "data.parquet",
  "algorithm": "kmeans",
  "params": {
    "n_clusters": [3, 5, 7, 10],
    "init": ["k-means++", "random"]
  },
  "metric": "attribute",
  "optimizer": "tpe"
}
```

### Graph/Network Configuration
```json
{
  "data_source": "pandas",
  "data_type": "graph", 
  "similarity": "network.edgelist",
  "algorithm": "louvain",
  "params": {
    "resolution": [0.5, 1.0, 1.5, 2.0]
  },
  "metric": "modularity",
  "optimizer": "grid"
}
```

### Attributed Graph Configuration
```json
{
  "data_source": "spark",
  "data_type": "attributed_graph",
  "features": "node_features.parquet",
  "similarity": "edges.parquet",
  "spark_config": {
    "spark.executor.memory": "4g",
    "spark.driver.memory": "2g"
  },
  "algorithm": "dmon",
  "params": {
    "num_clusters": [5, 10, 15, 20],
    "hidden_dim": [64, 128, 256]
  },
  "metric": "modularity",
  "optimizer": "tpe"
}
```

## Command Line Usage

```bash
# Get comprehensive help
python main.py -h

# List all available algorithms and metrics
python main.py -l

# Algorithm-specific help
python main.py kmeans -h

# Debug mode
python main.py --debug config.json
```

## Project Structure

```
Pattern/
├── core/                  # Core abstractions and factory patterns
│   ├── interfaces.py      # Abstract base classes
│   ├── factory.py         # Component factory
│   ├── api.py            # High-level API
│   └── logger.py         # Logging configuration
├── data/                 # Data loading (Pandas/Spark)
│   ├── loaders.py        # DataLoader implementations
│   └── utils.py          # Data utilities
├── models/               # Clustering algorithms
│   ├── attribute.py      # Feature-based models (KMeans, DBSCAN)
│   ├── network.py        # Graph-based models (Louvain, Spectral)
│   └── ag.py            # Attributed graph models (DMoN)
├── metrics/              # Evaluation metrics
│   ├── clustering_metrics.py  # Standard clustering metrics
│   └── quality.py        # Advanced quality measures
├── optimization/         # Hyperparameter optimization
│   └── strategies.py     # Grid, Random, TPE search
├── preprocessing/        # Data preprocessing
│   ├── normalizers.py    # Feature normalization
│   └── samplers.py       # Data sampling
├── visualization/        # Result visualization
│   ├── vis.py           # General plotting
│   ├── type_figs.py     # Data-type specific plots
│   └── mirkin_analysis.py  # Advanced analysis
├── stats/               # Statistical analysis
│   ├── stat.py          # Statistical computation
│   └── statanalyzer.py  # Analysis reporting
├── config/              # Configuration management
│   ├── registries.py    # Component registries
│   └── validator.py     # Config validation
├── cli/                 # Command line interface
│   └── parsers.py       # Argument parsing
├── main.py              # Application entry point
├── config*.json         # Example configurations
├── Test.ipynb           # Example notebook
└── cora.npz            # Sample dataset (Cora network)
```

## Advanced Features

### Spark Configuration
Customize Spark settings for large-scale processing:
```json
{
  "spark_config": {
    "spark.executor.memory": "8g",
    "spark.driver.memory": "4g",
    "spark.sql.adaptive.enabled": "true",
    "spark.sql.adaptive.coalescePartitions.enabled": "true"
  }
}
```

### Preprocessing Pipeline
Configure normalization and sampling:
```json
{
  "preprocessing": {
    "normalizer": {
      "methods": {
        "feature1": "zscore",
        "feature2": "minmax", 
        "feature3": "robust"
      }
    },
    "sampler": {
      "sample_size": 10000,
      "strategy": "random"
    }
  }
}
```

### Hyperparameter Optimization
Choose optimization strategy:
- **grid**: Exhaustive grid search
- **random**: Random parameter sampling  
- **tpe**: Tree-structured Parzen Estimator (recommended)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add your algorithm/metric following the interface patterns
4. Update documentation and tests
5. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

If you use Pattern in your research, please cite:
```bibtex
@software{pattern2024,
  title={Pattern: Scalable Unsupervised Learning for Multiple Data Types},
  author={Pattern Contributors},
  year={2024},
  url={https://github.com/Utopialvo/Pattern}
}
```