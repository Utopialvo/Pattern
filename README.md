# Pattern

**Library for scalable unsupervised learning**

## Description

A modular toolkit for unsupervised learning tasks with:
- Pandas & Apache Spark integration
- Extensible architecture for algorithms/metrics
- Hyperparameter optimization framework

## Features

- **Algorithms**:
- **Metrics**: WB, SW, Calinski-Harabasz, ANUI, AVU, AVI, modularity, density_modularity
- **Optimization**: Grid Search, Random Search
- **Data Formats**: Parquet, CSV, ORC (Pandas/Spark compatible)
- **Serialization**: Joblib/Pickle model persistence

## Requirements

- Python 3.8+
- PySpark 3.2.1+ (optional for Spark mode)
- Core Dependencies:
  - scikit-learn
  - pandas
  - numpy
  - joblib

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
├── main.py            # Entry point
├── README.md          # Project documentation
├── config.json        # Example configuration
└── Test.ipynb         # Example notebook
```

## Configuration Example

`config.json`:
```json
{
  "data_source": "pandas",
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