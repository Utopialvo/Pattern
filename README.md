# Pattern

**Library for scalable unsupervised learning**

## Description

Unsupervised learning library:
- Pandas & Apache Spark integration
- Extensible architecture for algorithms/metrics
- Hyperparameter optimization with optuna
- Extensible Metrics
- Visualization for interpretation result
- Statistic interpretation result

## Features

- **Algorithms**: KMeans, DBSCAN, Louvain, Spectral, Deep Modularity Network (DMoN)
- **Metrics**: WB, SW, Calinski-Harabasz, ANUI, AVU, AVI, modularity, density_modularity
- **Optimization**: Grid Search, Random Search, Tree-structured Parzen Estimator algorithm
- **Data Formats**: Parquet, CSV, ORC (Pandas/Spark compatible)
- **Serialization**: Joblib model persist
- **Visualization**: Graph and Features plots

## Requirements

- Python 3.11.10
- PySpark 3.3.1+ (optional for Spark mode)
- Core Dependencies:
  - scikit-learn=1.6.1
  - pandas=2.0.3
  - numpy=2.2.6
  - joblib=1.4.2
  - optuna=4.3.0
  - torch=2.7.0+cpu
  - networkx=3.4.1
  - torch_geometric=2.6.1
  - seaborn=0.13.2
  - matplotlib=3.10.3


## Installation

```bash
git clone https://github.com/Utopialvo/Pattern.git
cd Pattern
pip install -r requirements.txt
```

## Usage

### Run Pipeline

```bash
python main.py -c config.json
```

### Get Help

```bash
# Main help
python main.py -h

# List components
python main.py -l

# Algorithm-specific help
python main.py kmeans -h
```

## Project Structure

```
Pattern/
├── core/              # Base interfaces
├── data/              # Data loaders (Pandas/Spark)
├── models/            # Clustering implementations
├── metrics/           # Quality metrics
├── optimization/      # Hyperparameter strategies
├── preprocessing/     # Normalizers/Samplers
├── config/            # Configuration validation
├── cli/               # Command line interface
├── visualization/     # Result modeling visualization
├── stats/             # Cluster statistical analysis
├── main.py            # Entry point
├── README.md          # Project documentation
├── config.json        # Example configuration
├── cora.npz           # The Cora dataset consists of 2708 scientific publications classified into one of seven classes
└── Test.ipynb         # Example notebook
```

## Configuration Example

`config.json`:
```json
{
  "data_source": "pandas",
  "optimizer": "tpe",
  "plots_path":"results/datavis/kmeans",
  "stat_path":"results/stat/kmeans",
  "preprocessing": {
    "normalizer": {
      "methods": {
        "x1": "zscore",
        "x2": "range",
        "x3": "minmax"
      },
      "columns": [
        "x1",
        "x2",
        "x3"
      ]
    },
    "sampler": {
      "data_src": [
        "data.parquet",
        null
      ]
    }
  },
  "data_path": [
    "data.parquet",
    null
  ],
  "algorithm": "kmeans",
  "params": {
    "n_clusters": [
      3,
      5,
      7,
      10
    ],
    "init": [
      "k-means++",
      "random"
    ],
    "max_iter": [
      100,
      200
    ]
  },
  "metric": "attribute",
  "output_path": "best_kmeans.joblib"
}
```