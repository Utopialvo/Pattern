# Pattern

**Library for scalable unsupervised learning**

## Description

A modular toolkit for unsupervised learning tasks with:
- Pandas & Apache Spark integration
- Extensible architecture for algorithms/metrics
- Hyperparameter optimization framework

## Features

- **Algorithms**:
- **Metrics**: 
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
  "data_path": ["data.parquet"],
  "algorithm": "kmeans",
  "params": {
    "n_clusters": [2, 3, 4],
    "init": ["k-means++", "random"]
  },
  "metric": "silhouette",
  "output_path": "model.pkl"
}
```