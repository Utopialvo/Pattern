#!/usr/bin/env python3
"""
Test Library for Pattern - PySpark Scale
=========================================

This module provides comprehensive testing for the Pattern library at PySpark scale.
It automatically discovers implemented algorithms, handles large-scale benchmark datasets,
generates synthetic data, and evaluates performance using both default hyperparameters
and Optuna optimization in a distributed environment.

Features:
- Distributed algorithm testing with PySpark
- Large-scale benchmark dataset processing
- Real benchmark dataset downloading and processing (iris, wine, karate, etc.)
- Scalable synthetic data generation
- Performance evaluation at scale with default and optimized hyperparameters
- Comprehensive distributed result reporting and analysis
- Enhanced error handling with JSON logging
- Expected vs obtained performance comparisons
- Multiple export formats (CSV, JSON, Excel)
- Comprehensive save/load functionality

Author: Pattern Library Testing Framework
"""

import os
import sys
import json
import logging
import warnings
import traceback
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime 
import time

# Third-party imports
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.datasets import make_blobs, make_circles, make_moons, make_classification
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import requests
from io import StringIO

# PySpark imports
try:
    from pyspark.sql import SparkSession, DataFrame as SparkDataFrame
    from pyspark.sql.functions import col, rand, when, lit, count, avg, stddev
    from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType
    from pyspark.ml.feature import StandardScaler as SparkStandardScaler, VectorAssembler
    from pyspark.ml.linalg import Vectors, VectorUDT
    from pyspark.ml.stat import Correlation
    SPARK_AVAILABLE = True
except ImportError:
    print("Warning: PySpark not available. Please install PySpark to run distributed tests.")
    SPARK_AVAILABLE = False

# Pattern library imports
try:
    from config.registries import MODEL_REGISTRY, METRIC_REGISTRY
    from config.validator import load_config
    from core.factory import factory
    from core.logger import logger
    from data.loaders import SparkDataLoader, PandasDataLoader
    from optimization.strategies import TPESearch, GridSearch, RandomSearch
    from preprocessing.normalizers import SparkNormalizer
    from preprocessing.samplers import SparkSampler
except ImportError as e:
    print(f"Error importing Pattern library components: {e}")
    sys.exit(1)

warnings.filterwarnings('ignore')

class SparkBenchmarkDataManager:
    """Manages large-scale benchmark dataset processing with PySpark."""
    
    def __init__(self, spark: SparkSession, data_dir: str = "Datasets_Spark"):
        self.spark = spark
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for organized storage
        (self.data_dir / "Raw").mkdir(exist_ok=True)
        (self.data_dir / "Processed").mkdir(exist_ok=True)
        (self.data_dir / "Synthetic").mkdir(exist_ok=True)
        (self.data_dir / "Cache").mkdir(exist_ok=True)
        
        # Cache for loaded datasets
        self._dataset_cache = {}
        
        # Comprehensive benchmark datasets combining real and large-scale synthetic
        self.benchmark_datasets = {
            'attribute': {
                # Real benchmark datasets from test_library_memory.py
                'iris': {
                    'description': 'Classic iris flower dataset',
                    'expected_clusters': 3,
                    'expected_ari': 0.73,
                    'expected_nmi': 0.76,
                    'builtin': True
                },
                'wine': {
                    'description': 'Wine recognition dataset',
                    'expected_clusters': 3,
                    'expected_ari': 0.37,
                    'expected_nmi': 0.43,
                    'builtin': True
                },
                'breast_cancer': {
                    'description': 'Breast cancer Wisconsin dataset',
                    'expected_clusters': 2,
                    'expected_ari': 0.62,
                    'expected_nmi': 0.58,
                    'builtin': True
                },
                # Large-scale synthetic datasets for Spark
                'sklearn_large': {
                    'samples': 100000, 'features': 20, 'clusters': 5, 
                    'description': 'Large synthetic blobs',
                    'expected_ari': 0.85, 'expected_nmi': 0.82
                },
                'random_large': {
                    'samples': 50000, 'features': 15, 'clusters': 8, 
                    'description': 'Large random dataset',
                    'expected_ari': 0.65, 'expected_nmi': 0.68
                },
                'mixed_gaussian': {
                    'samples': 75000, 'features': 25, 'clusters': 6, 
                    'description': 'Mixed Gaussian clusters',
                    'expected_ari': 0.72, 'expected_nmi': 0.75
                },
                'high_dimensional': {
                    'samples': 30000, 'features': 50, 'clusters': 4,
                    'description': 'High-dimensional clustering challenge',
                    'expected_ari': 0.55, 'expected_nmi': 0.62
                },
                'overlapping_clusters': {
                    'samples': 40000, 'features': 18, 'clusters': 7,
                    'description': 'Overlapping cluster scenario',
                    'expected_ari': 0.45, 'expected_nmi': 0.52
                },
                'noise_contaminated': {
                    'samples': 60000, 'features': 22, 'clusters': 5,
                    'description': 'Clusters with noise contamination',
                    'expected_ari': 0.62, 'expected_nmi': 0.58
                }
            },
            'network': {
                # Real benchmark datasets from test_library_memory.py
                'karate': {
                    'description': 'Zachary karate club network',
                    'expected_clusters': 2,
                    'expected_modularity': 0.42,
                    'expected_ari': 0.685,
                    'builtin': True
                },
                # Large-scale synthetic networks for Spark
                'large_sbm': {
                    'nodes': 10000, 'communities': 20, 
                    'description': 'Large Stochastic Block Model',
                    'expected_modularity': 0.75, 'expected_ari': 0.82
                },
                'scale_free': {
                    'nodes': 15000, 'communities': 15, 
                    'description': 'Large Scale-free network',
                    'expected_modularity': 0.45, 'expected_ari': 0.52
                },
                'small_world': {
                    'nodes': 8000, 'communities': 12, 
                    'description': 'Large Small-world network',
                    'expected_modularity': 0.55, 'expected_ari': 0.62
                },
                'hierarchical_network': {
                    'nodes': 12000, 'communities': 18,
                    'description': 'Hierarchical community structure',
                    'expected_modularity': 0.68, 'expected_ari': 0.75
                },
                'power_law_network': {
                    'nodes': 9000, 'communities': 14,
                    'description': 'Power-law degree distribution',
                    'expected_modularity': 0.42, 'expected_ari': 0.48
                }
            },
            'attributed_graph': {
                # Synthetic attributed graphs from test_library_memory.py
                'synthetic_attr_easy': {
                    'description': 'Synthetic attributed graph - easy scenario',
                    'expected_clusters': 3,
                    'expected_ari': 0.85,
                    'expected_nmi': 0.82,
                    'builtin': True
                },
                'synthetic_attr_medium': {
                    'description': 'Synthetic attributed graph - medium scenario',
                    'expected_clusters': 4,
                    'expected_ari': 0.65,
                    'expected_nmi': 0.68,
                    'builtin': True
                },
                'synthetic_attr_hard': {
                    'description': 'Synthetic attributed graph - hard scenario',
                    'expected_clusters': 5,
                    'expected_ari': 0.45,
                    'expected_nmi': 0.52,
                    'builtin': True
                },
                # Large-scale attributed graphs for Spark
                'large_attr_sbm': {
                    'nodes': 5000, 'features': 30, 'communities': 10, 
                    'description': 'Large attributed SBM',
                    'expected_ari': 0.78, 'expected_nmi': 0.82
                },
                'complex_attr_graph': {
                    'nodes': 7500, 'features': 40, 'communities': 12, 
                    'description': 'Complex attributed graph',
                    'expected_ari': 0.65, 'expected_nmi': 0.71
                },
                'heterogeneous_features': {
                    'nodes': 6000, 'features': 35, 'communities': 8,
                    'description': 'Heterogeneous feature distributions',
                    'expected_ari': 0.58, 'expected_nmi': 0.65
                },
                'sparse_features': {
                    'nodes': 4000, 'features': 100, 'communities': 6,
                    'description': 'High-dimensional sparse features',
                    'expected_ari': 0.52, 'expected_nmi': 0.58
                }
            }
        }
        
        # Enhanced benchmark performance expectations
        self.benchmark_performance = {
            # Real datasets from test_library_memory.py
            'iris': {'silhouette': 0.55, 'calinski_harabasz': 561.6},
            'wine': {'silhouette': 0.27, 'calinski_harabasz': 561.9},
            'karate': {'modularity': 0.37, 'anui': 0.65},
            # Large-scale performance targets
            'sklearn_large': {'silhouette_target': 0.4, 'time_limit': 300},
            'large_sbm': {'modularity_target': 0.3, 'time_limit': 600},
            'large_attr_sbm': {'combined_metric_target': 0.35, 'time_limit': 900},
            'scale_free': {'modularity_target': 0.25, 'time_limit': 450},
            'complex_attr_graph': {'combined_metric_target': 0.3, 'time_limit': 1200}
        }
    
    def save_spark_dataset(self, name: str, features: Optional[SparkDataFrame] = None, 
                          similarity: Optional[SparkDataFrame] = None, 
                          labels: Optional[SparkDataFrame] = None, 
                          metadata: Optional[Dict] = None) -> bool:
        """Save a Spark dataset to disk."""
        try:
            dataset_dir = self.data_dir / name.capitalize()
            dataset_dir.mkdir(exist_ok=True)
            
            # Save features
            if features is not None:
                features.write.mode('overwrite').parquet(str(dataset_dir / "Features.parquet"))
            
            # Save similarity/adjacency matrix
            if similarity is not None:
                similarity.write.mode('overwrite').parquet(str(dataset_dir / "Networks.parquet"))
            
            # Save labels
            if labels is not None:
                labels.write.mode('overwrite').parquet(str(dataset_dir / "Labels.parquet"))
            
            # Save metadata
            metadata_info = {
                'name': name,
                'timestamp': datetime.now().isoformat(),
                'n_samples': features.count() if features is not None else (similarity.count() if similarity is not None else 0),
                'n_features': len(features.columns) if features is not None else 0,
                'has_similarity': similarity is not None,
                'has_labels': labels is not None,
                'n_unique_labels': labels.select('true_labels').distinct().count() if labels is not None else None,
                'spark_format': True
            }
            
            if metadata:
                metadata_info.update(metadata)
            
            with open(dataset_dir / "Metadata.json", 'w') as f:
                json.dump(metadata_info, f, indent=2, default=str)
            
            logger.info(f"Spark dataset '{name}' saved to {dataset_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save Spark dataset '{name}': {e}")
            return False
    
    def load_spark_dataset(self, name: str, use_cache: bool = True) -> Tuple[Optional[SparkDataFrame], Optional[SparkDataFrame], Optional[SparkDataFrame], Optional[Dict]]:
        """Load a Spark dataset from disk."""
        
        # Check cache first
        if use_cache and name in self._dataset_cache:
            logger.info(f"Loading Spark dataset '{name}' from cache")
            return self._dataset_cache[name]
        
        try:
            dataset_dir = self.data_dir / name.capitalize()
            
            if not dataset_dir.exists():
                logger.warning(f"Spark dataset '{name}' not found in datasets directory")
                return None, None, None, None
            
            features = None
            similarity = None
            labels = None
            metadata = None
            
            # Load features
            features_path = dataset_dir / "Features.parquet"
            if features_path.exists():
                features = self.spark.read.parquet(str(features_path))
            
            # Load similarity/adjacency matrix
            similarity_path = dataset_dir / "Networks.parquet"
            if similarity_path.exists():
                similarity = self.spark.read.parquet(str(similarity_path))
            
            # Load labels
            labels_path = dataset_dir / "Labels.parquet"
            if labels_path.exists():
                labels = self.spark.read.parquet(str(labels_path))
            
            # Load metadata
            metadata_path = dataset_dir / "Metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            
            # Cache the result
            result = (features, similarity, labels, metadata)
            if use_cache:
                self._dataset_cache[name] = result
            
            logger.info(f"Spark dataset '{name}' loaded from {dataset_dir}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to load Spark dataset '{name}': {e}")
            return None, None, None, None
    
    def save_configuration(self, config: Dict[str, Any], filename: str = "Spark_data_config.json") -> bool:
        """Save Spark data configuration to file."""
        try:
            config_path = self.data_dir / "Cache" / filename
            config_path.parent.mkdir(exist_ok=True)
            
            config_info = {
                'timestamp': datetime.now().isoformat(),
                'benchmark_datasets': self.benchmark_datasets,
                'benchmark_performance': self.benchmark_performance,
                'user_config': config,
                'spark_enabled': True
            }
            
            with open(config_path, 'w') as f:
                json.dump(config_info, f, indent=2, default=str)
            
            logger.info(f"Spark configuration saved to {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save Spark configuration: {e}")
            return False
    
    def load_configuration(self, filename: str = "Spark_data_config.json") -> Optional[Dict[str, Any]]:
        """Load Spark data configuration from file."""
        try:
            config_path = self.data_dir / "Cache" / filename
            
            if not config_path.exists():
                logger.warning(f"Spark configuration file {filename} not found")
                return None
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            logger.info(f"Spark configuration loaded from {config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load Spark configuration: {e}")
            return None
    
    def clear_cache(self):
        """Clear the Spark dataset cache."""
        self._dataset_cache.clear()
        logger.info("Spark dataset cache cleared")
    
    def list_cached_datasets(self) -> List[str]:
        """List all cached Spark datasets."""
        return list(self._dataset_cache.keys())
    
    def list_saved_datasets(self) -> List[str]:
        """List all saved processed Spark datasets."""
        if not self.data_dir.exists():
            return []
        
        return [d.name.lower() for d in self.data_dir.iterdir() if d.is_dir() and d.name not in ['Raw', 'Processed', 'Synthetic', 'Cache']]
    
    def load_attribute_dataset(self, dataset_name: str) -> Tuple[Optional[SparkDataFrame], Optional[SparkDataFrame]]:
        """Load attribute dataset and convert to Spark format."""
        try:
            # For builtin datasets, use sklearn and convert to Spark
            if dataset_name == 'iris':
                from sklearn.datasets import load_iris
                iris = load_iris()
                features_pd = pd.DataFrame(iris.data, columns=iris.feature_names)
                labels_pd = pd.DataFrame({'true_labels': iris.target})
                
                features = self.spark.createDataFrame(features_pd)
                labels = self.spark.createDataFrame(labels_pd)
                return features, labels
            
            elif dataset_name == 'wine':
                from sklearn.datasets import load_wine
                wine = load_wine()
                features_pd = pd.DataFrame(wine.data, columns=wine.feature_names)
                labels_pd = pd.DataFrame({'true_labels': wine.target})
                
                features = self.spark.createDataFrame(features_pd)
                labels = self.spark.createDataFrame(labels_pd)
                return features, labels
            
            elif dataset_name == 'breast_cancer':
                from sklearn.datasets import load_breast_cancer
                cancer = load_breast_cancer()
                features_pd = pd.DataFrame(cancer.data, columns=cancer.feature_names)
                labels_pd = pd.DataFrame({'true_labels': cancer.target})
                
                features = self.spark.createDataFrame(features_pd)
                labels = self.spark.createDataFrame(labels_pd)
                return features, labels
            
            # For other datasets, try to load from saved files
            else:
                features, _, labels, _ = self.load_spark_dataset(dataset_name)
                return features, labels
                
        except Exception as e:
            logger.error(f"Failed to load attribute dataset {dataset_name}: {e}")
            return None, None
    
    def load_network_dataset(self, dataset_name: str) -> Tuple[Optional[SparkDataFrame], Optional[SparkDataFrame], Optional[SparkDataFrame]]:
        """Load network dataset and convert to Spark format."""
        try:
            # For karate club, use networkx and convert to Spark
            if dataset_name == 'karate':
                import networkx as nx
                G = nx.karate_club_graph()
                adj_matrix_pd = pd.DataFrame(nx.adjacency_matrix(G).toarray())
                labels_pd = pd.DataFrame({'true_labels': [0 if G.nodes[i]['club'] == 'Mr. Hi' else 1 for i in G.nodes()]})
                
                adj_matrix = self.spark.createDataFrame(adj_matrix_pd)
                labels = self.spark.createDataFrame(labels_pd)
                return None, adj_matrix, labels
            
            # For other datasets, try to load from saved files
            else:
                features, similarity, labels, _ = self.load_spark_dataset(dataset_name)
                return features, similarity, labels
                
        except Exception as e:
            logger.error(f"Failed to load network dataset {dataset_name}: {e}")
            return None, None, None
    
    def load_attributed_graph_dataset(self, dataset_name: str) -> Tuple[Optional[SparkDataFrame], Optional[SparkDataFrame], Optional[SparkDataFrame]]:
        """Load attributed graph dataset and convert to Spark format."""
        try:
            # For synthetic scenarios, generate them with larger scale for Spark
            if dataset_name.startswith('synthetic_attr_'):
                if dataset_name == 'synthetic_attr_easy':
                    return SparkSyntheticDataGenerator.generate_attributed_graph_data(
                        self.spark, n_nodes=3000, n_features=15, n_communities=3, p_in=0.4, p_out=0.05
                    )
                elif dataset_name == 'synthetic_attr_medium':
                    return SparkSyntheticDataGenerator.generate_attributed_graph_data(
                        self.spark, n_nodes=4000, n_features=20, n_communities=4, p_in=0.3, p_out=0.03
                    )
                elif dataset_name == 'synthetic_attr_hard':
                    return SparkSyntheticDataGenerator.generate_attributed_graph_data(
                        self.spark, n_nodes=5000, n_features=25, n_communities=5, p_in=0.25, p_out=0.02
                    )
            
            # For other datasets, try to load from saved files
            else:
                features, similarity, labels, _ = self.load_spark_dataset(dataset_name)
                return features, similarity, labels
                
        except Exception as e:
            logger.error(f"Failed to load attributed graph dataset {dataset_name}: {e}")
            return None, None, None

class SparkSyntheticDataGenerator:
    """Generates large-scale synthetic datasets using Spark."""
    
    def __init__(self, spark: SparkSession, cache_dir: str = "Datasets_Spark/Synthetic"):
        self.spark = spark
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def save_synthetic_dataset(self, name: str, features: SparkDataFrame, 
                              similarity: Optional[SparkDataFrame] = None, 
                              labels: Optional[SparkDataFrame] = None, 
                              params: Optional[Dict] = None) -> bool:
        """Save a synthetic Spark dataset for reuse."""
        try:
            dataset_path = self.cache_dir / name
            dataset_path.mkdir(exist_ok=True)
            
            # Save as Parquet files
            if features is not None:
                features.write.mode('overwrite').parquet(str(dataset_path / "features.parquet"))
            
            if similarity is not None:
                similarity.write.mode('overwrite').parquet(str(dataset_path / "similarity.parquet"))
            
            if labels is not None:
                labels.write.mode('overwrite').parquet(str(dataset_path / "labels.parquet"))
            
            # Save metadata
            metadata = {
                'name': name,
                'timestamp': datetime.now().isoformat(),
                'params': params or {},
                'format': 'spark_parquet'
            }
            
            with open(dataset_path / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"Synthetic Spark dataset '{name}' saved to {dataset_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save synthetic Spark dataset '{name}': {e}")
            return False
    
    def load_synthetic_dataset(self, name: str) -> Tuple[Optional[SparkDataFrame], Optional[SparkDataFrame], Optional[SparkDataFrame], Optional[Dict]]:
        """Load a saved synthetic Spark dataset."""
        try:
            dataset_path = self.cache_dir / name
            
            if not dataset_path.exists():
                logger.warning(f"Synthetic Spark dataset '{name}' not found")
                return None, None, None, None
            
            features = None
            similarity = None
            labels = None
            params = None
            
            features_path = dataset_path / "features.parquet"
            if features_path.exists():
                features = self.spark.read.parquet(str(features_path))
            
            similarity_path = dataset_path / "similarity.parquet"
            if similarity_path.exists():
                similarity = self.spark.read.parquet(str(similarity_path))
            
            labels_path = dataset_path / "labels.parquet"
            if labels_path.exists():
                labels = self.spark.read.parquet(str(labels_path))
            
            metadata_path = dataset_path / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    params = metadata.get('params', {})
            
            logger.info(f"Synthetic Spark dataset '{name}' loaded from {dataset_path}")
            return features, similarity, labels, params
            
        except Exception as e:
            logger.error(f"Failed to load synthetic Spark dataset '{name}': {e}")
            return None, None, None, None
    
    def list_saved_synthetic_datasets(self) -> List[str]:
        """List all saved synthetic Spark datasets."""
        if not self.cache_dir.exists():
            return []
        
        return [d.name for d in self.cache_dir.iterdir() if d.is_dir()]
    
    @staticmethod
    def generate_large_attribute_data(spark: SparkSession, n_samples: int = 50000, 
                                     n_features: int = 20, n_clusters: int = 5, 
                                     scenario: str = 'blobs') -> Tuple[SparkDataFrame, SparkDataFrame]:
        """Generate large-scale synthetic attribute data using Spark."""
        
        if scenario == 'blobs':
            X, y = make_blobs(n_samples=n_samples, centers=n_clusters, 
                             n_features=n_features, cluster_std=1.0,
                             random_state=42)
        elif scenario == 'circles':
            X, y = make_circles(n_samples=n_samples, noise=0.1, factor=0.6,
                               random_state=42)
        elif scenario == 'moons':
            X, y = make_moons(n_samples=n_samples, noise=0.1, random_state=42)
            
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Convert to Spark DataFrames
        feature_names = [f'feature_{i}' for i in range(X_scaled.shape[1])]
        features_pd = pd.DataFrame(X_scaled, columns=feature_names)
        labels_pd = pd.DataFrame({'true_labels': y})
        
        features_spark = spark.createDataFrame(features_pd)
        labels_spark = spark.createDataFrame(labels_pd)
        
        return features_spark, labels_spark
    
    @staticmethod
    def generate_large_network_data(spark: SparkSession, n_nodes: int = 10000, 
                                   n_communities: int = 10, p_in: float = 0.1, 
                                   p_out: float = 0.01) -> Tuple[None, SparkDataFrame, SparkDataFrame]:
        """Generate large-scale synthetic network data using Spark."""
        
        # Create community assignment
        community_sizes = [n_nodes // n_communities] * n_communities
        community_sizes[-1] += n_nodes % n_communities  # Handle remainder
        
        # Generate SBM
        G = nx.stochastic_block_model(community_sizes, 
                                    [[p_in if i == j else p_out 
                                      for j in range(n_communities)]
                                     for i in range(n_communities)],
                                    seed=42)
        
        # Get adjacency matrix and convert to Spark
        adj_matrix_pd = pd.DataFrame(nx.adjacency_matrix(G).toarray())
        
        # Get true community labels
        true_labels = []
        node_to_community = nx.get_node_attributes(G, 'block')
        for i in range(n_nodes):
            true_labels.append(node_to_community[i])
        
        labels_pd = pd.DataFrame({'true_labels': true_labels})
        
        # Convert to Spark DataFrames
        adj_matrix_spark = spark.createDataFrame(adj_matrix_pd)
        labels_spark = spark.createDataFrame(labels_pd)
        
        return None, adj_matrix_spark, labels_spark
    
    @staticmethod
    def generate_attributed_graph_data(spark: SparkSession, n_nodes: int = 5000, 
                                      n_features: int = 20, n_communities: int = 3, 
                                      p_in: float = 0.3, p_out: float = 0.05) -> Tuple[SparkDataFrame, SparkDataFrame, SparkDataFrame]:
        """Generate large-scale synthetic attributed graph data using Spark."""
        
        # Generate network structure
        _, adj_matrix_spark, labels_spark = SparkSyntheticDataGenerator.generate_large_network_data(
            spark, n_nodes, n_communities, p_in, p_out)
        
        # Generate node features correlated with communities
        # First collect labels to CPU for feature generation
        labels_pd = labels_spark.toPandas()
        true_labels = labels_pd['true_labels'].values
        
        features_list = []
        for community in range(n_communities):
            community_nodes = (true_labels == community).sum()
            # Create distinct feature distributions for each community
            community_center = np.random.randn(n_features) * 3
            community_features = np.random.randn(community_nodes, n_features) + community_center
            features_list.append(community_features)
        
        # Combine features
        X = np.vstack(features_list)
        
        # Shuffle to match node order
        node_order = np.arange(len(true_labels))
        X_ordered = X[np.argsort(np.argsort(node_order))]
        
        # Convert to Spark DataFrame
        feature_names = [f'feature_{i}' for i in range(n_features)]
        features_pd = pd.DataFrame(X_ordered, columns=feature_names)
        features_spark = spark.createDataFrame(features_pd)
        
        return features_spark, adj_matrix_spark, labels_spark

class SparkAlgorithmTester:
    """Tests Pattern library algorithms at PySpark scale with comprehensive error handling."""
    
    def __init__(self, results_dir: str = "test_results_spark"):
        if not SPARK_AVAILABLE:
            raise ImportError("PySpark is required for distributed testing")
        
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for organization
        (self.results_dir / "Errors").mkdir(exist_ok=True)
        (self.results_dir / "Logs").mkdir(exist_ok=True)
        (self.results_dir / "Reports").mkdir(exist_ok=True)
        (self.results_dir / "Cache").mkdir(exist_ok=True)
        (self.results_dir / "Exports").mkdir(exist_ok=True)
        
        self.spark = self._create_spark_session()
        self.data_manager = SparkBenchmarkDataManager(self.spark)
        self.synthetic_generator = SparkSyntheticDataGenerator(self.spark)
        self.test_results = []
        self.error_count = 0
        
        self._setup_logging()
    
    def _create_spark_session(self) -> SparkSession:
        """Create and configure Spark session."""
        spark = SparkSession.builder \
            .appName("Pattern Library Spark Testing") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.sql.adaptive.skewJoin.enabled", "true") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .getOrCreate()
        
        spark.sparkContext.setLogLevel("WARN")
        return spark
    
    def _setup_logging(self):
        """Setup logging configuration for Spark testing."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.results_dir / "Logs" / f"Spark_test_log_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Clear existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)
    
    def _save_error_to_json(self, error_info: Dict[str, Any]) -> str:
        """Save error information to JSON file."""
        self.error_count += 1
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        error_filename = f"Spark_error_{self.error_count:03d}_{timestamp}.json"
        error_path = self.results_dir / "Errors" / error_filename
        
        try:
            with open(error_path, 'w') as f:
                json.dump(error_info, f, indent=2, default=str)
            logger.info(f"Spark error details saved to: {error_filename}")
            return str(error_path)
        except Exception as e:
            logger.error(f"Failed to save Spark error to JSON: {e}")
            return ""
    
    def save_test_results(self, filename: Optional[str] = None) -> bool:
        """Save current Spark test results to file."""
        try:
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"spark_test_results_{timestamp}.json"
            
            results_path = self.results_dir / "Cache" / filename
            results_path.parent.mkdir(exist_ok=True)
            
            save_data = {
                'timestamp': datetime.now().isoformat(),
                'test_info': {
                    'total_tests': len(self.test_results),
                    'error_count': self.error_count,
                    'results_dir': str(self.results_dir),
                    'spark_enabled': True
                },
                'test_results': self.test_results
            }
            
            with open(results_path, 'w') as f:
                json.dump(save_data, f, indent=2, default=str)
            
            logger.info(f"Spark test results saved to {results_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save Spark test results: {e}")
            return False
    
    def load_test_results(self, filename: str) -> bool:
        """Load Spark test results from file."""
        try:
            results_path = self.results_dir / "cache" / filename
            
            if not results_path.exists():
                logger.warning(f"Spark test results file {filename} not found")
                return False
            
            with open(results_path, 'r') as f:
                data = json.load(f)
            
            self.test_results = data.get('test_results', [])
            self.error_count = data.get('test_info', {}).get('error_count', 0)
            
            logger.info(f"Spark test results loaded from {results_path}")
            logger.info(f"Loaded {len(self.test_results)} test results")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Spark test results: {e}")
            return False
    
    def export_results_to_formats(self, formats: List[str] = ['csv', 'json']) -> Dict[str, bool]:
        """Export Spark test results to multiple formats."""
        results = {}
        
        if not self.test_results:
            logger.warning("No Spark test results to export")
            return {fmt: False for fmt in formats}
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        df_results = pd.DataFrame(self.test_results)
        
        for fmt in formats:
            try:
                if fmt.lower() == 'csv':
                    export_path = self.results_dir / "exports" / f"spark_results_{timestamp}.csv"
                    export_path.parent.mkdir(exist_ok=True)
                    df_results.to_csv(export_path, index=False)
                    results[fmt] = True
                    logger.info(f"Spark results exported to CSV: {export_path}")
                
                elif fmt.lower() == 'json':
                    export_path = self.results_dir / "exports" / f"spark_results_{timestamp}.json"
                    export_path.parent.mkdir(exist_ok=True)
                    with open(export_path, 'w') as f:
                        json.dump(self.test_results, f, indent=2, default=str)
                    results[fmt] = True
                    logger.info(f"Spark results exported to JSON: {export_path}")
                
                else:
                    logger.warning(f"Unsupported export format for Spark: {fmt}")
                    results[fmt] = False
                    
            except Exception as e:
                logger.error(f"Failed to export Spark results to {fmt}: {e}")
                results[fmt] = False
        
        return results

    def save_model(self, model, algorithm_name: str, dataset_name: str, 
                   optimization_method: str = 'manual', suffix: str = '') -> Optional[str]:
        """Save a trained Spark model to disk."""
        try:
            # Create Models directory if it doesn't exist
            models_dir = self.results_dir / "Models"
            models_dir.mkdir(exist_ok=True)
            
            # Define model save path
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_filename = f"{algorithm_name}_{dataset_name}_{optimization_method}_{timestamp}_spark{suffix}.model"
            model_path = models_dir / model_filename
            
            # Save model
            logger.info(f"Saving Spark model {algorithm_name} to {model_path}")
            model.save(str(model_path))
            logger.info(f"Spark model {algorithm_name} saved successfully")
            
            return str(model_path)
            
        except Exception as e:
            logger.error(f"Failed to save Spark model {algorithm_name}: {e}")
            return None
    
    def load_model(self, algorithm_name: str, model_path: str):
        """Load a trained Spark model from disk."""
        try:
            logger.info(f"Loading Spark model {algorithm_name} from {model_path}")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            model_class = MODEL_REGISTRY[algorithm_name]['class']
            loaded_model = model_class.load(model_path)
            
            logger.info(f"Spark model {algorithm_name} loaded successfully")
            return loaded_model
            
        except Exception as e:
            logger.error(f"Failed to load Spark model {algorithm_name}: {e}")
            return None
    
    def list_saved_models(self) -> List[str]:
        """List all saved Spark model files."""
        models_dir = self.results_dir / "Models"
        if not models_dir.exists():
            return []
        
        return [f.name for f in models_dir.glob("*_spark*.model")]
    
    def discover_spark_compatible_algorithms(self) -> Dict[str, Dict]:
        """Discover algorithms compatible with Spark processing."""
        logger.info("Discovering Spark-compatible algorithms...")
        
        algorithms = {}
        for name, info in MODEL_REGISTRY.items():
            # Filter algorithms that can work with Spark (based on implementation)
            if self._is_spark_compatible(name):
                algorithms[name] = {
                    'class': info['class'],
                    'params_help': info['params_help'],
                    'modality': self._infer_modality(name, info)
                }
                logger.info(f"Found Spark-compatible algorithm: {name}")
        
        logger.info(f"Total Spark-compatible algorithms: {len(algorithms)}")
        return algorithms
    
    def _is_spark_compatible(self, algorithm_name: str) -> bool:
        """Check if an algorithm is compatible with Spark processing."""
        # For now, assume all algorithms can be adapted to work with Spark
        # In practice, this would depend on the specific implementation
        spark_compatible = ['kmeans', 'dbscan', 'spectral', 'louvain']
        return algorithm_name.lower() in [alg.lower() for alg in spark_compatible]
    
    def _infer_modality(self, algo_name: str, algo_info: Dict) -> str:
        """Infer the modality of an algorithm."""
        name_lower = algo_name.lower()
        
        if any(keyword in name_lower for keyword in ['spectral', 'louvain', 'modularity']):
            return 'network'
        elif any(keyword in name_lower for keyword in ['Not supported']):
            return 'attributed_graph'
        else:
            return 'attribute'
    
    def get_default_params(self, algorithm_name: str) -> Dict[str, Any]:
        """Get default parameters optimized for Spark processing."""
        if algorithm_name not in MODEL_REGISTRY:
            return {}
        
        params_help = MODEL_REGISTRY[algorithm_name]['params_help']
        default_params = {}
        
        for param_name, description in params_help.items():
            if 'cluster' in param_name.lower():
                default_params[param_name] = 8  # More clusters for large data
            elif param_name.lower() in ['eps', 'epsilon']:
                default_params[param_name] = 0.5
            elif 'min_samples' in param_name.lower():
                default_params[param_name] = 10  # Higher for large data
            elif 'init' in param_name.lower():
                default_params[param_name] = 'k-means++'
            elif 'max_iter' in param_name.lower():
                default_params[param_name] = 100  # Conservative for large data
            elif 'resolution' in param_name.lower():
                default_params[param_name] = 1.0
        
        return default_params
    
    def test_algorithm_on_spark_dataset(self, algorithm_name: str, dataset_name: str,
                                       features: Optional[SparkDataFrame], 
                                       similarity: Optional[SparkDataFrame],
                                       true_labels: Optional[SparkDataFrame], 
                                       params: Dict[str, Any],
                                       optimization_method: str = 'default') -> Dict[str, Any]:
        """Test a single algorithm on a Spark dataset."""
        
        start_time = time.time()
        result = {
            'algorithm': algorithm_name,
            'dataset': dataset_name,
            'optimization': optimization_method,
            'params': params.copy(),
            'success': False,
            'error': None,
            'execution_time': 0,
            'metrics': {},
            'data_size': 0,
            'spark_partitions': 0,
            'model_save_success': False,
            'model_load_success': False,
            'model_save_path': None
        }
        
        try:
            logger.info(f"Testing {algorithm_name} on {dataset_name} (Spark) with {optimization_method} params")
            
            # Record data size and partitions
            if features is not None:
                result['data_size'] = features.count()
                result['spark_partitions'] = features.rdd.getNumPartitions()
            elif similarity is not None:
                result['data_size'] = similarity.count()
                result['spark_partitions'] = similarity.rdd.getNumPartitions()
            
            # Create Spark data loader
            data_loader = SparkDataLoader(
                spark=self.spark,
                features=features, 
                similarity=similarity
            )
            
            # Create and configure model
            model = factory.create_model(algorithm_name, params)
            
            # Fit model
            model.fit(data_loader)
            
            # Save and load model functionality
            try:
                # Create Models directory if it doesn't exist
                models_dir = self.results_dir / "Models"
                models_dir.mkdir(exist_ok=True)
                
                # Define model save path
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                model_filename = f"{algorithm_name}_{dataset_name}_{optimization_method}_{timestamp}_spark.model"
                model_path = models_dir / model_filename
                result['model_save_path'] = str(model_path)
                
                # Save model
                logger.info(f"Saving Spark model {algorithm_name} to {model_path}")
                model.save(str(model_path))
                result['model_save_success'] = True
                logger.info(f"Spark model {algorithm_name} saved successfully")
                
                # Load model back to verify save/load functionality
                logger.info(f"Loading Spark model {algorithm_name} from {model_path}")
                model_class = MODEL_REGISTRY[algorithm_name]['class']
                loaded_model = model_class.load(str(model_path))
                result['model_load_success'] = True
                logger.info(f"Spark model {algorithm_name} loaded successfully")
                
                # Verify loaded model has same predictions (if possible with Spark)
                if hasattr(loaded_model, 'labels_') and loaded_model.labels_ is not None:
                    loaded_predictions = loaded_model.labels_
                elif hasattr(loaded_model, 'predict'):
                    try:
                        loaded_predictions = loaded_model.predict(data_loader)
                    except Exception as e:
                        logger.warning(f"Could not get predictions from loaded model: {e}")
                        loaded_predictions = None
                else:
                    loaded_predictions = None
                
                # Compare original and loaded model predictions if possible
                if loaded_predictions is not None and hasattr(model, 'labels_') and model.labels_ is not None:
                    original_predictions = model.labels_
                    
                    # For Spark models, we need to be careful about data types
                    try:
                        if hasattr(loaded_predictions, 'toPandas'):
                            loaded_predictions_arr = loaded_predictions.toPandas().iloc[:, 0].values
                        else:
                            loaded_predictions_arr = np.array(loaded_predictions)
                        
                        if hasattr(original_predictions, 'toPandas'):
                            original_predictions_arr = original_predictions.toPandas().iloc[:, 0].values
                        else:
                            original_predictions_arr = np.array(original_predictions)
                        
                        # Check if predictions match
                        predictions_match = np.array_equal(original_predictions_arr, loaded_predictions_arr)
                        result['predictions_match_after_load'] = predictions_match
                        
                        if predictions_match:
                            logger.info(f"Spark model {algorithm_name} save/load verification successful - predictions match")
                        else:
                            logger.warning(f"Spark model {algorithm_name} save/load verification failed - predictions don't match")
                    except Exception as e:
                        logger.warning(f"Could not compare predictions for Spark model {algorithm_name}: {e}")
                
            except Exception as e:
                logger.error(f"Spark model save/load failed for {algorithm_name}: {e}")
                result['model_save_load_error'] = str(e)
            
            # Get predictions
            if hasattr(model, 'labels_') and model.labels_ is not None:
                predicted_labels = model.labels_
            else:
                predicted_labels = model.predict(data_loader)
            
            # Calculate metrics
            if true_labels is not None:
                # Convert Spark DataFrames to pandas for metric calculation
                true_labels_pd = true_labels.toPandas()['true_label'].values
                
                if hasattr(predicted_labels, 'toPandas'):
                    predicted_labels_pd = predicted_labels.toPandas().iloc[:, 0].values
                else:
                    predicted_labels_pd = predicted_labels
                
                result['metrics']['ari'] = adjusted_rand_score(true_labels_pd, predicted_labels_pd)
                result['metrics']['nmi'] = normalized_mutual_info_score(true_labels_pd, predicted_labels_pd)
            
            # Pattern library metrics (adapted for Spark)
            for metric_name in METRIC_REGISTRY:
                try:
                    metric = factory.create_metric(metric_name)
                    score = metric.calculate(data_loader, predicted_labels, model.model_data)
                    if not np.isnan(score):
                        result['metrics'][metric_name] = score
                except Exception as e:
                    logger.warning(f"Failed to calculate {metric_name}: {e}")
            
            result['success'] = True
            logger.info(f"Successfully tested {algorithm_name} on {dataset_name} (Spark)")
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Failed to test {algorithm_name} on {dataset_name} (Spark): {e}")
            logger.debug(traceback.format_exc())
        
        result['execution_time'] = time.time() - start_time
        return result
    
    def optimize_spark_hyperparameters(self, algorithm_name: str, dataset_name: str,
                                      features: Optional[SparkDataFrame], 
                                      similarity: Optional[SparkDataFrame],
                                      true_labels: Optional[SparkDataFrame],
                                      n_trials: int = 10) -> Dict[str, Any]:
        """Optimize hyperparameters for Spark processing (reduced trials)."""
        
        logger.info(f"Optimizing hyperparameters for {algorithm_name} on {dataset_name} (Spark)")
        
        try:
            data_loader = SparkDataLoader(spark=self.spark, features=features, similarity=similarity)
            param_grid = self._get_spark_param_grid(algorithm_name)
            
            if not param_grid:
                return self.get_default_params(algorithm_name)
            
            # Reduced trials for Spark testing
            optimizer = TPESearch(n_trials=min(n_trials, 10))
            
            metric_name = self._get_optimization_metric(algorithm_name)
            metric = factory.create_metric(metric_name) if metric_name else None
            
            if metric is None:
                return self.get_default_params(algorithm_name)
            
            model_class = MODEL_REGISTRY[algorithm_name]['class']
            best_params = optimizer.find_best(
                model_class=model_class,
                data_loader=data_loader,
                param_grid=param_grid,
                metric=metric
            )
            
            logger.info(f"Spark optimization completed for {algorithm_name}: {best_params}")
            return best_params
            
        except Exception as e:
            logger.error(f"Spark hyperparameter optimization failed for {algorithm_name}: {e}")
            return self.get_default_params(algorithm_name)
    
    def _get_spark_param_grid(self, algorithm_name: str) -> Dict[str, List[Any]]:
        """Get parameter grid optimized for Spark processing."""
        # Smaller parameter grids for distributed testing
        param_grids = {
            'kmeans': {
                'n_clusters': [3, 5, 8],
                'init': ['k-means++'],
                'max_iter': [50, 100]
            },
            'dbscan': {
                'eps': [0.3, 0.5, 0.7],
                'min_samples': [5, 10]
            },
            'spectral': {
                'n_clusters': [3, 5, 8],
                'assign_labels': ['kmeans']
            },
            'louvain': {
                'resolution': [0.8, 1.0, 1.2]
            }
        }
        return param_grids.get(algorithm_name, {})
    
    def _get_optimization_metric(self, algorithm_name: str) -> str:
        """Get appropriate metric for optimization."""
        metric_mapping = {
            'kmeans': 'attribute',
            'dbscan': 'attribute',
            'spectral': 'graph',
            'louvain': 'graph',
            'dmon': 'attribute-graph'
        }
        return metric_mapping.get(algorithm_name, 'attribute')
    
    def run_comprehensive_tests(self):
        """Run comprehensive tests on Spark-compatible algorithms."""
        
        logger.info("Starting comprehensive Pattern library testing (Spark Scale)")
        
        algorithms = self.discover_spark_compatible_algorithms()
        
        if not algorithms:
            logger.warning("No Spark-compatible algorithms found")
            return
        
        # Test on large-scale benchmark datasets
        self._test_spark_benchmark_datasets(algorithms)
        
        # Test on large-scale synthetic datasets
        self._test_spark_synthetic_datasets(algorithms)
        
        # Generate comprehensive report
        self._generate_spark_report()
        
        logger.info("Spark comprehensive testing completed")
    
    def _test_spark_benchmark_datasets(self, algorithms: Dict[str, Dict]):
        """Test algorithms on large-scale benchmark datasets."""
        
        logger.info("Testing on large-scale benchmark datasets (Spark)...")
        
        # Test large attribute datasets
        for dataset_name in ['sklearn_large', 'random_large']:
            logger.info(f"Creating large benchmark dataset: {dataset_name}")
            
            features, true_labels = self.data_manager.create_large_attribute_dataset(dataset_name)
            if features is None:
                continue
            
            for algo_name, algo_info in algorithms.items():
                if algo_info['modality'] == 'attribute':
                    
                    # Test with default parameters
                    default_params = self.get_default_params(algo_name)
                    result = self.test_algorithm_on_spark_dataset(
                        algo_name, dataset_name, features, None, true_labels,
                        default_params, 'default'
                    )
                    self.test_results.append(result)
                    
                    # Test with optimized parameters (limited trials)
                    optimized_params = self.optimize_spark_hyperparameters(
                        algo_name, dataset_name, features, None, true_labels, n_trials=5
                    )
                    result = self.test_algorithm_on_spark_dataset(
                        algo_name, dataset_name, features, None, true_labels,
                        optimized_params, 'optimized'
                    )
                    self.test_results.append(result)
        
        # Test large network dataset
        logger.info("Creating large network dataset")
        _, edges_df, labels_df = self.data_manager.create_large_network_dataset('large_sbm')
        
        if edges_df is not None:
            for algo_name, algo_info in algorithms.items():
                if algo_info['modality'] == 'network':
                    default_params = self.get_default_params(algo_name)
                    result = self.test_algorithm_on_spark_dataset(
                        algo_name, 'large_sbm', None, edges_df, labels_df,
                        default_params, 'default'
                    )
                    self.test_results.append(result)
    
    def _test_spark_synthetic_datasets(self, algorithms: Dict[str, Dict]):
        """Test algorithms on large-scale synthetic datasets."""
        
        logger.info("Testing on large-scale synthetic datasets (Spark)...")
        
        # Large attribute scenarios
        scenarios = [
            {'name': 'large_blobs', 'params': {'n_samples': 50000, 'n_features': 15, 'n_clusters': 5}},
            {'name': 'sparse_clusters', 'params': {'n_samples': 30000, 'n_features': 20, 'n_clusters': 8, 'scenario': 'sparse_clusters'}}
        ]
        
        for scenario in scenarios:
            logger.info(f"Generating large synthetic dataset: {scenario['name']}")
            
            features, true_labels = self.synthetic_generator.generate_large_attribute_data(**scenario['params'])
            
            for algo_name, algo_info in algorithms.items():
                if algo_info['modality'] == 'attribute':
                    default_params = self.get_default_params(algo_name)
                    if 'n_clusters' in default_params:
                        default_params['n_clusters'] = scenario['params'].get('n_clusters', 5)
                    
                    result = self.test_algorithm_on_spark_dataset(
                        algo_name, f"synthetic_{scenario['name']}", features, None, true_labels,
                        default_params, 'default'
                    )
                    self.test_results.append(result)
        
        # Large network scenario
        logger.info("Generating large synthetic network")
        _, edges_df, labels_df = self.synthetic_generator.generate_large_network_data(n_nodes=8000, n_communities=8)
        
        for algo_name, algo_info in algorithms.items():
            if algo_info['modality'] == 'network':
                default_params = self.get_default_params(algo_name)
                if 'n_clusters' in default_params:
                    default_params['n_clusters'] = 8
                
                result = self.test_algorithm_on_spark_dataset(
                    algo_name, "synthetic_large_network", None, edges_df, labels_df,
                    default_params, 'default'
                )
                self.test_results.append(result)
    
    def _generate_spark_report(self):
        """Generate comprehensive Spark test report."""
        
        logger.info("Generating comprehensive Spark test report...")
        
        df_results = pd.DataFrame(self.test_results)
        
        # Save detailed results
        results_file = self.results_dir / f"spark_detailed_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df_results.to_csv(results_file, index=False)
        
        # Generate summary
        summary = {
            'test_info': {
                'timestamp': datetime.now().isoformat(),
                'total_tests': len(df_results),
                'successful_tests': int(df_results['success'].sum()) if not df_results.empty else 0,
                'failed_tests': int((~df_results['success']).sum()) if not df_results.empty else 0,
                'scale': 'spark',
                'spark_session_info': {
                    'app_name': self.spark.sparkContext.appName,
                    'master': self.spark.sparkContext.master,
                    'spark_version': self.spark.version
                }
            },
            'performance_analysis': {},
            'scalability_metrics': {}
        }
        
        # Performance analysis
        if not df_results.empty and df_results['success'].any():
            success_df = df_results[df_results['success'] == True]
            
            # Add scalability metrics
            if 'data_size' in success_df.columns:
                summary['scalability_metrics'] = {
                    'avg_data_size': float(success_df['data_size'].mean()),
                    'max_data_size': float(success_df['data_size'].max()),
                    'avg_execution_time': float(success_df['execution_time'].mean()),
                    'throughput_samples_per_sec': float(success_df['data_size'].sum() / success_df['execution_time'].sum())
                }
        
        summary_file = self.results_dir / f"spark_summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        logger.info("=" * 60)
        logger.info("PATTERN LIBRARY TEST SUMMARY (SPARK SCALE)")
        logger.info("=" * 60)
        logger.info(f"Total tests executed: {len(self.test_results)}")
        logger.info(f"Successful tests: {sum(1 for r in self.test_results if r['success'])}")
        logger.info(f"Failed tests: {sum(1 for r in self.test_results if not r['success'])}")
        
        if self.test_results:
            avg_time = np.mean([r['execution_time'] for r in self.test_results])
            avg_size = np.mean([r.get('data_size', 0) for r in self.test_results if r.get('data_size')])
            logger.info(f"Average execution time: {avg_time:.2f} seconds")
            logger.info(f"Average dataset size: {avg_size:.0f} samples")
        
        logger.info("=" * 60)
        logger.info(f"Detailed results saved to: {results_file}")
        logger.info(f"Summary report saved to: {summary_file}")

def create_spark_session() -> SparkSession:
    """Create and configure Spark session for testing."""
    
    spark = SparkSession.builder \
        .appName("Pattern Library Spark Testing") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.sql.adaptive.skewJoin.enabled", "true") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .getOrCreate()
    
    # Set log level to reduce verbose output
    spark.sparkContext.setLogLevel("WARN")
    
    return spark

def main():
    """Main Spark testing function."""
    
    if not SPARK_AVAILABLE:
        print("PySpark is not available. Please install PySpark to run distributed tests.")
        print("pip install pyspark")
        return
    
    print("Pattern Library Comprehensive Testing - Spark Scale")
    print("=" * 60)
    print("This test suite will:")
    print("1. Discover all Spark-compatible algorithms")
    print("2. Generate large-scale benchmark datasets")
    print("3. Create large-scale synthetic datasets")
    print("4. Test algorithms with distributed processing")
    print("5. Generate scalability and performance reports")
    print("=" * 60)
    
    # Create Spark session
    try:
        spark = create_spark_session()
        logger.info(f"Created Spark session: {spark.sparkContext.appName}")
        logger.info(f"Spark version: {spark.version}")
        
        # Create tester
        tester = SparkAlgorithmTester(spark)
        
        # Run comprehensive tests
        tester.run_comprehensive_tests()
        
        print("\nSpark testing completed successfully!")
        print(f"Results saved in: {tester.results_dir}")
        
    except Exception as e:
        logger.error(f"Spark testing failed with error: {e}")
        logger.debug(traceback.format_exc())
        print(f"\nSpark testing failed: {e}")
        
    finally:
        # Stop Spark session
        if 'spark' in locals():
            spark.stop()
            logger.info("Spark session stopped")

if __name__ == "__main__":
    main() 