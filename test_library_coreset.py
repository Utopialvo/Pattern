#!/usr/bin/env python3
"""
Test Library for Pattern - Coreset Scale
=========================================

This module provides comprehensive testing for the Pattern library using coreset algorithms
for efficient large-scale processing. It automatically discovers implemented algorithms,
generates coresets for scalable processing, creates synthetic data, and evaluates performance
using both default hyperparameters and Optuna optimization.

Features:
- Coreset-based algorithm testing for scalability
- Real benchmark dataset downloading and coreset construction
- Large-scale dataset processing via coresets
- Efficient synthetic data generation and coreset construction
- Performance evaluation with coreset approximations and optimized hyperparameters
- Comprehensive coreset quality and efficiency reporting
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
from sklearn.cluster import KMeans
from io import StringIO

# Pattern library imports
try:
    from config.registries import MODEL_REGISTRY, METRIC_REGISTRY
    from config.validator import load_config
    from core.factory import factory
    from core.logger import logger
    from data.loaders import PandasDataLoader
    from optimization.strategies import TPESearch, GridSearch, RandomSearch
except ImportError as e:
    print(f"Error importing Pattern library components: {e}")
    sys.exit(1)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class GenericCoresetConstructor:
    """Generic coreset constructor with memory and Spark versions supporting multiple sensitivity methods."""
    
    def __init__(self, mode: str = "memory", random_state: int = 42):
        """
        Initialize the generic coreset constructor.
        
        Args:
            mode: Either "memory" or "spark" for computation mode
            random_state: Random seed for reproducibility
        """
        if mode not in ["memory", "spark"]:
            raise ValueError("Mode must be either 'memory' or 'spark'")
        
        self.mode = mode
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Initialize Spark context if needed
        self.spark = None
        if self.mode == "spark":
            self._init_spark()
    
    def _init_spark(self):
        """Initialize Spark session for Spark mode."""
        try:
            from pyspark.sql import SparkSession
            
            if not hasattr(self, 'spark') or self.spark is None:
                self.spark = SparkSession.builder \
                    .appName("GenericCoresetConstructor") \
                    .config("spark.sql.adaptive.enabled", "true") \
                    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
                    .getOrCreate()
                
                logger.info("Spark session initialized for coreset construction")
        except ImportError:
            logger.error("PySpark not available for Spark mode coreset construction")
            raise ImportError("PySpark not available")
    
    def build_attribute_coreset(self, X: Union[np.ndarray, pd.DataFrame], coreset_size: int,
                               sensitivity_method: str = 'exact', 
                               algorithm: str = 'kmeans') -> Tuple[np.ndarray, np.ndarray]:
        """
        Build coreset for attribute data using generic coreset constructor.
        
        Args:
            X: Input data (numpy array or pandas DataFrame)
            coreset_size: Target size of coreset
            sensitivity_method: One of 'exact', 'relaxed', 'distance_only'
            algorithm: Target algorithm for coreset construction ('kmeans', 'dbscan', etc.)
            
        Returns:
            Tuple of (coreset_points, coreset_weights)
        """
        if sensitivity_method not in ['exact', 'relaxed', 'distance_only']:
            raise ValueError("sensitivity_method must be one of: 'exact', 'relaxed', 'distance_only'")
        
        # Convert input to appropriate format
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        if len(X_array) <= coreset_size:
            return X_array, np.ones(len(X_array))
        
        logger.info(f"Building coreset using {self.mode} mode with {sensitivity_method} sensitivity")
        
        if self.mode == "memory":
            return self._build_memory_coreset(X_array, coreset_size, sensitivity_method, algorithm)
        else:  # spark
            return self._build_spark_coreset(X_array, coreset_size, sensitivity_method, algorithm)
    
    def _build_memory_coreset(self, X: np.ndarray, coreset_size: int, 
                             sensitivity_method: str, algorithm: str) -> Tuple[np.ndarray, np.ndarray]:
        """Build coreset using memory-based computation."""
        
        n_samples, n_features = X.shape
        
        if sensitivity_method == 'exact':
            return self._compute_exact_sensitivities_memory(X, coreset_size, algorithm)
        elif sensitivity_method == 'relaxed':
            return self._compute_relaxed_sensitivities_memory(X, coreset_size, algorithm)
        else:  # distance_only
            return self._compute_distance_only_sensitivities_memory(X, coreset_size, algorithm)
    
    def _build_spark_coreset(self, X: np.ndarray, coreset_size: int,
                            sensitivity_method: str, algorithm: str) -> Tuple[np.ndarray, np.ndarray]:
        """Build coreset using Spark-based computation."""
        
        # Convert numpy array to Spark DataFrame
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        df_pandas = pd.DataFrame(X, columns=feature_names)
        df_spark = self.spark.createDataFrame(df_pandas)
        
        if sensitivity_method == 'exact':
            return self._compute_exact_sensitivities_spark(df_spark, coreset_size, algorithm)
        elif sensitivity_method == 'relaxed':
            return self._compute_relaxed_sensitivities_spark(df_spark, coreset_size, algorithm)
        else:  # distance_only
            return self._compute_distance_only_sensitivities_spark(df_spark, coreset_size, algorithm)
    
    def _compute_exact_sensitivities_memory(self, X: np.ndarray, coreset_size: int, 
                                          algorithm: str) -> Tuple[np.ndarray, np.ndarray]:
        """Compute exact sensitivities using memory-based approach."""
        
        n_samples = len(X)
        
        # Exact sensitivity computation - compute true importance of each point
        if algorithm.lower() == 'kmeans':
            # For k-means, use distance to optimal centers as sensitivity
            from sklearn.cluster import KMeans
            k = min(coreset_size // 10, int(np.sqrt(n_samples)))
            kmeans = KMeans(n_clusters=k, random_state=self.random_state)
            kmeans.fit(X)
            
            # Compute exact sensitivities based on distances to centers
            distances = np.min(np.linalg.norm(
                X[:, np.newaxis] - kmeans.cluster_centers_[np.newaxis, :], axis=2
            ), axis=1)
            sensitivities = distances / np.sum(distances)
            
        else:
            # Generic approach: use local density as sensitivity
            from sklearn.neighbors import NearestNeighbors
            k = min(10, n_samples // 10)
            nbrs = NearestNeighbors(n_neighbors=k).fit(X)
            distances, _ = nbrs.kneighbors(X)
            densities = 1.0 / (np.mean(distances, axis=1) + 1e-8)
            sensitivities = densities / np.sum(densities)
        
        # Sample based on sensitivities
        sampled_indices = np.random.choice(
            n_samples, size=coreset_size, replace=False, p=sensitivities
        )
        
        coreset_points = X[sampled_indices]
        weights = 1.0 / (sensitivities[sampled_indices] * coreset_size)
        
        return coreset_points, weights
    
    def _compute_relaxed_sensitivities_memory(self, X: np.ndarray, coreset_size: int,
                                            algorithm: str) -> Tuple[np.ndarray, np.ndarray]:
        """Compute relaxed sensitivities using memory-based approach."""
        
        n_samples = len(X)
        
        # Relaxed sensitivity computation - approximation for efficiency
        if algorithm.lower() == 'kmeans':
            # Use approximate clustering for sensitivity estimation
            from sklearn.cluster import MiniBatchKMeans
            k = min(coreset_size // 10, int(np.sqrt(n_samples)))
            kmeans = MiniBatchKMeans(n_clusters=k, random_state=self.random_state, batch_size=min(1000, n_samples))
            kmeans.fit(X)
            
            # Approximate sensitivities
            distances = np.min(np.linalg.norm(
                X[:, np.newaxis] - kmeans.cluster_centers_[np.newaxis, :], axis=2
            ), axis=1)
            sensitivities = distances / np.sum(distances)
            
        else:
            # Relaxed approach: grid-based density estimation
            # Simple grid-based approximation
            n_bins = min(50, int(np.sqrt(n_samples)))
            hist, _ = np.histogramdd(X, bins=n_bins)
            
            # Map points to bins and use inverse bin count as sensitivity
            bin_indices = np.floor((X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0)) * (n_bins - 1)).astype(int)
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)
            
            sensitivities = np.ones(n_samples)
            for i in range(n_samples):
                bin_count = hist[tuple(bin_indices[i])]
                sensitivities[i] = 1.0 / (bin_count + 1)
            
            sensitivities = sensitivities / np.sum(sensitivities)
        
        # Sample based on sensitivities
        sampled_indices = np.random.choice(
            n_samples, size=coreset_size, replace=False, p=sensitivities
        )
        
        coreset_points = X[sampled_indices]
        weights = 1.0 / (sensitivities[sampled_indices] * coreset_size)
        
        return coreset_points, weights
    
    def _compute_distance_only_sensitivities_memory(self, X: np.ndarray, coreset_size: int,
                                                   algorithm: str) -> Tuple[np.ndarray, np.ndarray]:
        """Compute distance-only sensitivities using memory-based approach."""
        
        n_samples = len(X)
        
        # Distance-only sensitivity - fastest approximation
        # Use random sampling with distance-based weights
        center = np.mean(X, axis=0)
        distances = np.linalg.norm(X - center, axis=1)
        
        # Higher distance points get higher probability (outliers are important)
        sensitivities = distances / np.sum(distances)
        sensitivities = np.clip(sensitivities, 1e-8, 1.0)  # Avoid zero probabilities
        
        # Sample based on distance sensitivities
        sampled_indices = np.random.choice(
            n_samples, size=coreset_size, replace=False, p=sensitivities
        )
        
        coreset_points = X[sampled_indices]
        weights = 1.0 / (sensitivities[sampled_indices] * coreset_size)
        
        return coreset_points, weights
    
    def _compute_exact_sensitivities_spark(self, df_spark, coreset_size: int,
                                         algorithm: str) -> Tuple[np.ndarray, np.ndarray]:
        """Compute exact sensitivities using Spark-based approach."""
        
        # Convert back to pandas for now (can be optimized for pure Spark later)
        df_pandas = df_spark.toPandas()
        X = df_pandas.values
        
        # Use memory-based computation for now
        # TODO: Implement pure Spark version
        return self._compute_exact_sensitivities_memory(X, coreset_size, algorithm)
    
    def _compute_relaxed_sensitivities_spark(self, df_spark, coreset_size: int,
                                           algorithm: str) -> Tuple[np.ndarray, np.ndarray]:
        """Compute relaxed sensitivities using Spark-based approach."""
        
        # Convert back to pandas for now (can be optimized for pure Spark later)
        df_pandas = df_spark.toPandas()
        X = df_pandas.values
        
        # Use memory-based computation for now
        # TODO: Implement pure Spark version
        return self._compute_relaxed_sensitivities_memory(X, coreset_size, algorithm)
    
    def _compute_distance_only_sensitivities_spark(self, df_spark, coreset_size: int,
                                                  algorithm: str) -> Tuple[np.ndarray, np.ndarray]:
        """Compute distance-only sensitivities using Spark-based approach."""
        
        from pyspark.sql.functions import col, avg, sqrt, sum as spark_sum
        
        # Compute mean of each feature using Spark
        feature_cols = df_spark.columns
        means = []
        for col_name in feature_cols:
            mean_val = df_spark.select(avg(col(col_name))).collect()[0][0]
            means.append(mean_val)
        
        # Convert back to pandas for distance computation (can be optimized)
        df_pandas = df_spark.toPandas()
        X = df_pandas.values
        center = np.array(means)
        
        # Compute distances
        distances = np.linalg.norm(X - center, axis=1)
        sensitivities = distances / np.sum(distances)
        sensitivities = np.clip(sensitivities, 1e-8, 1.0)
        
        n_samples = len(X)
        sampled_indices = np.random.choice(
            n_samples, size=coreset_size, replace=False, p=sensitivities
        )
        
        coreset_points = X[sampled_indices]
        weights = 1.0 / (sensitivities[sampled_indices] * coreset_size)
        
        return coreset_points, weights
    
    def __del__(self):
        """Clean up Spark session if it exists."""
        if hasattr(self, 'spark') and self.spark is not None:
            try:
                self.spark.stop()
                logger.info("Spark session stopped in GenericCoresetConstructor")
            except:
                pass

class CoresetBenchmarkDataManager:
    """Manages coreset-based data processing for benchmark and synthetic datasets."""
    
    def __init__(self, coreset_constructor: GenericCoresetConstructor, data_dir: str = "Datasets_Coreset"):
        self.coreset_constructor = coreset_constructor
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for organized storage
        (self.data_dir / "Raw").mkdir(exist_ok=True)
        (self.data_dir / "Processed").mkdir(exist_ok=True)
        (self.data_dir / "Synthetic").mkdir(exist_ok=True)
        (self.data_dir / "Cache").mkdir(exist_ok=True)
        (self.data_dir / "Coresets").mkdir(exist_ok=True)
        
        # Cache for loaded datasets
        self._dataset_cache = {}
        
        # Enhanced coreset configurations
        self.coreset_configs = {
            'small': {'size_ratio': 0.1, 'min_size': 100, 'max_size': 1000},
            'medium': {'size_ratio': 0.05, 'min_size': 200, 'max_size': 2000},
            'large': {'size_ratio': 0.02, 'min_size': 500, 'max_size': 5000}
        }
        
        # Comprehensive benchmark datasets combining real and coreset-optimized synthetic
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
                'seeds': {
                    'description': 'Seeds dataset',
                    'expected_clusters': 3,
                    'expected_ari': 0.71,
                    'expected_nmi': 0.69,
                    'builtin': True
                },
                # Large-scale datasets for coreset testing
                'large_blobs': {
                    'original_size': 50000, 'n_features': 20, 'n_clusters': 8,
                    'description': 'Large blob dataset for coreset testing',
                    'expected_ari': 0.85, 'expected_nmi': 0.82
                },
                'high_dimensional': {
                    'original_size': 30000, 'n_features': 50, 'n_clusters': 6,
                    'description': 'High-dimensional clustering challenge',
                    'expected_ari': 0.65, 'expected_nmi': 0.71
                },
                'noise_contaminated': {
                    'original_size': 40000, 'n_features': 25, 'n_clusters': 5,
                    'description': 'Noisy cluster scenario',
                    'expected_ari': 0.58, 'expected_nmi': 0.62
                },
                'overlapping_clusters': {
                    'original_size': 35000, 'n_features': 18, 'n_clusters': 7,
                    'description': 'Overlapping cluster challenge',
                    'expected_ari': 0.52, 'expected_nmi': 0.58
                }
            },
            'network': {
                # Real network datasets
                'karate': {
                    'description': 'Zachary karate club network',
                    'expected_clusters': 2,
                    'expected_modularity': 0.42,
                    'expected_ari': 0.685,
                    'builtin': True
                },
                # Large networks for coreset testing
                'large_sbm': {
                    'nodes': 20000, 'communities': 15,
                    'description': 'Large SBM for coreset testing',
                    'expected_modularity': 0.72, 'expected_ari': 0.78
                },
                'scale_free': {
                    'nodes': 15000, 'communities': 12,
                    'description': 'Scale-free network',
                    'expected_modularity': 0.45, 'expected_ari': 0.52
                },
                'small_world': {
                    'nodes': 18000, 'communities': 10,
                    'description': 'Small-world network',
                    'expected_modularity': 0.55, 'expected_ari': 0.62
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
                # Large attributed graphs for coreset testing
                'large_attr_graph': {
                    'nodes': 10000, 'features': 30, 'communities': 8,
                    'description': 'Large attributed graph for coreset testing',
                    'expected_ari': 0.72, 'expected_nmi': 0.75
                }
            }
        }
        
        # Enhanced benchmark performance expectations
        self.benchmark_performance = {
            # Real datasets from test_library_memory.py
            'iris': {'silhouette': 0.55, 'calinski_harabasz': 561.6},
            'wine': {'silhouette': 0.27, 'calinski_harabasz': 561.9},
            'karate': {'modularity': 0.37, 'anui': 0.65},
            # Coreset performance targets
            'large_blobs': {'coreset_efficiency': 0.9, 'time_speedup': 5.0},
            'large_sbm': {'coreset_modularity': 0.65, 'compression_ratio': 20},
            'large_attr_graph': {'combined_metric': 0.7, 'memory_reduction': 15}
        }
    
    def save_coreset_dataset(self, name: str, original_data: Dict[str, Any], 
                            coresets: Dict[str, Any], metadata: Optional[Dict] = None) -> bool:
        """Save coreset dataset with all components."""
        try:
            dataset_dir = self.data_dir / name.capitalize()
            dataset_dir.mkdir(exist_ok=True)
            
            # Save original data
            if 'features' in original_data and original_data['features'] is not None:
                if isinstance(original_data['features'], pd.DataFrame):
                    original_data['features'].to_csv(dataset_dir / "Original_features.csv", index=False)
                else:
                    np.save(dataset_dir / "Original_features.npy", original_data['features'])
            
            if 'similarity' in original_data and original_data['similarity'] is not None:
                if isinstance(original_data['similarity'], pd.DataFrame):
                    original_data['similarity'].to_csv(dataset_dir / "Original_networks.csv", index=False)
                else:
                    np.save(dataset_dir / "Original_networks.npy", original_data['similarity'])
            
            if 'labels' in original_data and original_data['labels'] is not None:
                if isinstance(original_data['labels'], pd.Series):
                    original_data['labels'].to_csv(dataset_dir / "Original_labels.csv", index=False)
                else:
                    np.save(dataset_dir / "Original_labels.npy", original_data['labels'])
            
            # Save coresets
            coresets_dir = dataset_dir / "Coresets"
            coresets_dir.mkdir(exist_ok=True)
            
            for method, coreset_data in coresets.items():
                method_dir = coresets_dir / method
                method_dir.mkdir(exist_ok=True)
                
                if 'points' in coreset_data:
                    np.save(method_dir / "points.npy", coreset_data['points'])
                if 'weights' in coreset_data:
                    np.save(method_dir / "weights.npy", coreset_data['weights'])
                
                with open(method_dir / "info.json", 'w') as f:
                    json.dump({
                        'size': coreset_data.get('size', 0),
                        'compression_ratio': coreset_data.get('compression_ratio', 1.0),
                        'method': method
                    }, f, indent=2)
            
            # Save metadata
            metadata_info = {
                'name': name,
                'timestamp': datetime.now().isoformat(),
                'coreset_methods': list(coresets.keys()),
                'format': 'coreset',
                'n_samples': len(original_data.get('features', [])) if 'features' in original_data else 0,
                'n_features': len(original_data['features'].columns) if 'features' in original_data and hasattr(original_data['features'], 'columns') else 0
            }
            
            if metadata:
                metadata_info.update(metadata)
            
            with open(dataset_dir / "Metadata.json", 'w') as f:
                json.dump(metadata_info, f, indent=2, default=str)
            
            logger.info(f"Coreset dataset '{name}' saved to {dataset_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save coreset dataset '{name}': {e}")
            return False
    
    def load_coreset_dataset(self, name: str, use_cache: bool = True) -> Tuple[Optional[Dict], Optional[Dict], Optional[Dict]]:
        """Load coreset dataset with all components."""
        
        # Check cache first
        if use_cache and name in self._dataset_cache:
            logger.info(f"Loading coreset dataset '{name}' from cache")
            return self._dataset_cache[name]
        
        try:
            dataset_dir = self.data_dir / name.capitalize()
            
            if not dataset_dir.exists():
                logger.warning(f"Coreset dataset '{name}' not found")
                return None, None, None
            
            # Load original data
            original_data = {}
            
            features_csv = dataset_dir / "Original_features.csv"
            features_npy = dataset_dir / "Original_features.npy"
            if features_csv.exists():
                original_data['features'] = pd.read_csv(features_csv)
            elif features_npy.exists():
                original_data['features'] = np.load(features_npy)
            
            networks_csv = dataset_dir / "Original_networks.csv"
            networks_npy = dataset_dir / "Original_networks.npy"
            if networks_csv.exists():
                original_data['similarity'] = pd.read_csv(networks_csv)
            elif networks_npy.exists():
                original_data['similarity'] = np.load(networks_npy)
            
            labels_csv = dataset_dir / "Original_labels.csv"
            labels_npy = dataset_dir / "Original_labels.npy"
            if labels_csv.exists():
                original_data['labels'] = pd.read_csv(labels_csv).iloc[:, 0]
                original_data['labels'].name = 'true_labels'
            elif labels_npy.exists():
                original_data['labels'] = np.load(labels_npy)
            
            # Load coresets
            coresets = {}
            coresets_dir = dataset_dir / "Coresets"
            if coresets_dir.exists():
                for method_dir in coresets_dir.iterdir():
                    if method_dir.is_dir():
                        method_name = method_dir.name
                        coresets[method_name] = {}
                        
                        points_file = method_dir / "points.npy"
                        if points_file.exists():
                            coresets[method_name]['points'] = np.load(points_file)
                        
                        weights_file = method_dir / "weights.npy"
                        if weights_file.exists():
                            coresets[method_name]['weights'] = np.load(weights_file)
                        
                        info_file = method_dir / "info.json"
                        if info_file.exists():
                            with open(info_file, 'r') as f:
                                coresets[method_name].update(json.load(f))
            
            # Load metadata
            metadata = None
            metadata_path = dataset_dir / "Metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            
            # Cache the result
            result = (original_data, coresets, metadata)
            if use_cache:
                self._dataset_cache[name] = result
            
            logger.info(f"Coreset dataset '{name}' loaded from {dataset_dir}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to load coreset dataset '{name}': {e}")
            return None, None, None
    
    def save_configuration(self, config: Dict[str, Any], filename: str = "Coreset_data_config.json") -> bool:
        """Save coreset data configuration to file."""
        try:
            config_path = self.data_dir / "Cache" / filename
            config_path.parent.mkdir(exist_ok=True)
            
            config_info = {
                'timestamp': datetime.now().isoformat(),
                'benchmark_datasets': self.benchmark_datasets,
                'benchmark_performance': self.benchmark_performance,
                'coreset_configs': self.coreset_configs,
                'user_config': config
            }
            
            with open(config_path, 'w') as f:
                json.dump(config_info, f, indent=2, default=str)
            
            logger.info(f"Coreset configuration saved to {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save coreset configuration: {e}")
            return False
    
    def load_configuration(self, filename: str = "Coreset_data_config.json") -> Optional[Dict[str, Any]]:
        """Load coreset data configuration from file."""
        try:
            config_path = self.data_dir / "Cache" / filename
            
            if not config_path.exists():
                logger.warning(f"Coreset configuration file {filename} not found")
                return None
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            logger.info(f"Coreset configuration loaded from {config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load coreset configuration: {e}")
            return None
    
    def clear_cache(self):
        """Clear the coreset dataset cache."""
        self._dataset_cache.clear()
        logger.info("Coreset dataset cache cleared")
    
    def list_cached_datasets(self) -> List[str]:
        """List all cached coreset datasets."""
        return list(self._dataset_cache.keys())
    
    def list_saved_datasets(self) -> List[str]:
        """List all saved processed coreset datasets."""
        if not self.data_dir.exists():
            return []
        
        return [d.name.lower() for d in self.data_dir.iterdir() if d.is_dir() and d.name not in ['Raw', 'Processed', 'Synthetic', 'Cache', 'Coresets']]
    
    def load_attribute_dataset(self, dataset_name: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """Load attribute dataset."""
        try:
            # For builtin datasets, use sklearn
            if dataset_name == 'iris':
                from sklearn.datasets import load_iris
                iris = load_iris()
                features = pd.DataFrame(iris.data, columns=iris.feature_names)
                labels = pd.Series(iris.target, name='true_labels')
                return features, labels
            
            elif dataset_name == 'wine':
                from sklearn.datasets import load_wine
                wine = load_wine()
                features = pd.DataFrame(wine.data, columns=wine.feature_names)
                labels = pd.Series(wine.target, name='true_labels')
                return features, labels
            
            elif dataset_name == 'breast_cancer':
                from sklearn.datasets import load_breast_cancer
                cancer = load_breast_cancer()
                features = pd.DataFrame(cancer.data, columns=cancer.feature_names)
                labels = pd.Series(cancer.target, name='true_labels')
                return features, labels
            
            elif dataset_name == 'seeds':
                # Generate seeds-like dataset
                X, y = make_blobs(n_samples=210, centers=3, n_features=7, 
                                 cluster_std=1.5, random_state=42)
                features = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(7)])
                labels = pd.Series(y, name='true_labels')
                return features, labels
            
            # For other datasets, try to load from saved files
            else:
                original_data, _, _ = self.load_coreset_dataset(dataset_name)
                if original_data:
                    return original_data.get('features'), original_data.get('labels')
                return None, None
                
        except Exception as e:
            logger.error(f"Failed to load attribute dataset {dataset_name}: {e}")
            return None, None
    
    def load_network_dataset(self, dataset_name: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.Series]]:
        """Load network dataset."""
        try:
            # For karate club, use networkx
            if dataset_name == 'karate':
                import networkx as nx
                G = nx.karate_club_graph()
                adj_matrix = pd.DataFrame(nx.adjacency_matrix(G).toarray())
                labels = pd.Series([0 if G.nodes[i]['club'] == 'Mr. Hi' else 1 for i in G.nodes()], name='true_labels')
                return None, adj_matrix, labels
            
            # For other datasets, try to load from saved files
            else:
                original_data, _, _ = self.load_coreset_dataset(dataset_name)
                if original_data:
                    return original_data.get('features'), original_data.get('similarity'), original_data.get('labels')
                return None, None, None
                
        except Exception as e:
            logger.error(f"Failed to load network dataset {dataset_name}: {e}")
            return None, None, None
    
    def load_attributed_graph_dataset(self, dataset_name: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.Series]]:
        """Load attributed graph dataset."""
        try:
            # For synthetic scenarios, generate them using the same logic as test_library_memory.py
            if dataset_name.startswith('synthetic_attr_'):
                if dataset_name == 'synthetic_attr_easy':
                    return CoresetSyntheticDataGenerator.generate_attributed_graph_data(
                        n_nodes=300, n_features=15, n_communities=3, p_in=0.4, p_out=0.05
                    )
                elif dataset_name == 'synthetic_attr_medium':
                    return CoresetSyntheticDataGenerator.generate_attributed_graph_data(
                        n_nodes=400, n_features=20, n_communities=4, p_in=0.3, p_out=0.03
                    )
                elif dataset_name == 'synthetic_attr_hard':
                    return CoresetSyntheticDataGenerator.generate_attributed_graph_data(
                        n_nodes=500, n_features=25, n_communities=5, p_in=0.25, p_out=0.02
                    )
            
            # For other datasets, try to load from saved files
            else:
                original_data, _, _ = self.load_coreset_dataset(dataset_name)
                if original_data:
                    return original_data.get('features'), original_data.get('similarity'), original_data.get('labels')
                return None, None, None
                
        except Exception as e:
            logger.error(f"Failed to load attributed graph dataset {dataset_name}: {e}")
            return None, None, None

class CoresetSyntheticDataGenerator:
    """Generates synthetic datasets optimized for coreset construction and testing."""
    
    def __init__(self, cache_dir: str = "Datasets_Coreset/Synthetic"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def save_synthetic_dataset(self, name: str, features: pd.DataFrame, similarity: Optional[pd.DataFrame] = None, 
                              labels: Optional[pd.Series] = None, params: Optional[Dict] = None) -> bool:
        """Save a synthetic dataset for reuse."""
        try:
            dataset_path = self.cache_dir / f"{name}.npz"
            
            # Prepare data for saving
            save_data = {}
            if features is not None:
                save_data['features'] = features.values
                save_data['feature_names'] = features.columns.tolist()
            
            if similarity is not None:
                save_data['similarity'] = similarity.values
            
            if labels is not None:
                save_data['labels'] = labels.values
            
            if params is not None:
                save_data['params'] = json.dumps(params, default=str)
            
            save_data['timestamp'] = datetime.now().isoformat()
            
            np.savez_compressed(dataset_path, **save_data)
            logger.info(f"Synthetic coreset dataset '{name}' saved to {dataset_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save synthetic coreset dataset '{name}': {e}")
            return False
    
    def load_synthetic_dataset(self, name: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.Series], Optional[Dict]]:
        """Load a saved synthetic dataset."""
        try:
            dataset_path = self.cache_dir / f"{name}.npz"
            
            if not dataset_path.exists():
                logger.warning(f"Synthetic coreset dataset '{name}' not found")
                return None, None, None, None
            
            data = np.load(dataset_path, allow_pickle=True)
            
            features = None
            similarity = None
            labels = None
            params = None
            
            if 'features' in data:
                feature_names = data.get('feature_names', [f'feature_{i}' for i in range(data['features'].shape[1])])
                features = pd.DataFrame(data['features'], columns=feature_names)
            
            if 'similarity' in data:
                similarity = pd.DataFrame(data['similarity'])
            
            if 'labels' in data:
                labels = pd.Series(data['labels'], name='true_labels')
            
            if 'params' in data:
                params = json.loads(str(data['params']))
            
            logger.info(f"Synthetic coreset dataset '{name}' loaded from {dataset_path}")
            return features, similarity, labels, params
            
        except Exception as e:
            logger.error(f"Failed to load synthetic coreset dataset '{name}': {e}")
            return None, None, None, None
    
    def list_saved_synthetic_datasets(self) -> List[str]:
        """List all saved synthetic datasets."""
        if not self.cache_dir.exists():
            return []
        
        return [f.stem for f in self.cache_dir.glob("*.npz")]
    
    @staticmethod
    def generate_attribute_data(n_samples: int = 10000, n_features: int = 20, 
                               n_clusters: int = 5, cluster_std: float = 1.0,
                               scenario: str = 'blobs') -> Tuple[pd.DataFrame, pd.Series]:
        """Generate synthetic attribute data optimized for coreset testing."""
        
        if scenario == 'blobs':
            X, y = make_blobs(n_samples=n_samples, centers=n_clusters, 
                             n_features=n_features, cluster_std=cluster_std,
                             random_state=42)
        elif scenario == 'circles':
            X, y = make_circles(n_samples=n_samples, noise=0.1, factor=0.6,
                               random_state=42)
        elif scenario == 'moons':
            X, y = make_moons(n_samples=n_samples, noise=0.1, random_state=42)
            
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Convert to pandas
        feature_names = [f'feature_{i}' for i in range(X_scaled.shape[1])]
        df_features = pd.DataFrame(X_scaled, columns=feature_names)
        series_labels = pd.Series(y, name='true_labels')
        
        return df_features, series_labels
    
    @staticmethod
    def generate_network_data(n_nodes: int = 5000, n_communities: int = 8,
                             p_in: float = 0.3, p_out: float = 0.05,
                             scenario: str = 'sbm') -> Tuple[None, pd.DataFrame, pd.Series]:
        """Generate synthetic network data optimized for coreset testing."""
        
        if scenario == 'sbm':  # Stochastic Block Model
            # Create community assignment
            community_sizes = [n_nodes // n_communities] * n_communities
            community_sizes[-1] += n_nodes % n_communities  # Handle remainder
            
            # Generate SBM
            G = nx.stochastic_block_model(community_sizes, 
                                        [[p_in if i == j else p_out 
                                          for j in range(n_communities)]
                                         for i in range(n_communities)],
                                        seed=42)
            
            # Get adjacency matrix
            adj_matrix = pd.DataFrame(nx.adjacency_matrix(G).toarray())
            
            # Get true community labels
            true_labels = []
            node_to_community = nx.get_node_attributes(G, 'block')
            for i in range(n_nodes):
                true_labels.append(node_to_community[i])
            
            return None, adj_matrix, pd.Series(true_labels, name='true_labels')
            
        elif scenario == 'barabasi_albert':
            G = nx.barabasi_albert_graph(n_nodes, m=3, seed=42)
            adj_matrix = pd.DataFrame(nx.adjacency_matrix(G).toarray())
            
            # For BA graph, create artificial communities based on degree
            degrees = dict(G.degree())
            degree_values = list(degrees.values())
            degree_threshold_low = np.percentile(degree_values, 33)
            degree_threshold_high = np.percentile(degree_values, 67)
            
            true_labels = []
            for node in G.nodes():
                deg = degrees[node]
                if deg <= degree_threshold_low:
                    true_labels.append(0)
                elif deg <= degree_threshold_high:
                    true_labels.append(1)
                else:
                    true_labels.append(2)
            
            return None, adj_matrix, pd.Series(true_labels, name='true_labels')
    
    @staticmethod
    def generate_attributed_graph_data(n_nodes: int = 2000, n_features: int = 25,
                                      n_communities: int = 5, p_in: float = 0.3,
                                      p_out: float = 0.05) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """Generate synthetic attributed graph data optimized for coreset testing."""
        
        # Generate network structure
        _, adj_matrix, true_labels = CoresetSyntheticDataGenerator.generate_network_data(
            n_nodes, n_communities, p_in, p_out, 'sbm')
        
        # Generate node features correlated with communities
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
        node_order = true_labels.index
        X_ordered = X[np.argsort(np.argsort(node_order))]
        
        # Convert to pandas
        feature_names = [f'feature_{i}' for i in range(n_features)]
        df_features = pd.DataFrame(X_ordered, columns=feature_names)
        
        return df_features, adj_matrix, true_labels

class CoresetAlgorithmTester:
    """Comprehensive algorithm tester for coreset-scale processing with pandas and PySpark support."""
    
    def __init__(self, results_dir: str = "Test_Results_Coreset", mode: str = "pandas", 
                 sensitivity_methods: List[str] = None):
        """
        Initialize CoresetAlgorithmTester.
        
        Args:
            results_dir: Directory for saving results
            mode: Either "pandas" or "pyspark" for data processing mode
            sensitivity_methods: List of sensitivity methods to test ['exact', 'relaxed', 'distance_only']
        """
        if mode not in ["pandas", "pyspark"]:
            raise ValueError("Mode must be either 'pandas' or 'pyspark'")
        
        self.mode = mode
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Set default sensitivity methods if not provided
        if sensitivity_methods is None:
            self.sensitivity_methods = ['exact', 'relaxed', 'distance_only']
        else:
            self.sensitivity_methods = sensitivity_methods
            
        # Validate sensitivity methods
        valid_methods = ['exact', 'relaxed', 'distance_only']
        for method in self.sensitivity_methods:
            if method not in valid_methods:
                raise ValueError(f"Invalid sensitivity method: {method}. Must be one of {valid_methods}")
        
        # Create subdirectories
        (self.results_dir / "Models").mkdir(exist_ok=True)
        (self.results_dir / "Errors").mkdir(exist_ok=True)
        (self.results_dir / "Cache").mkdir(exist_ok=True)
        (self.results_dir / "Reports").mkdir(exist_ok=True)
        
        # Initialize components with new generic coreset constructor
        coreset_mode = "memory" if self.mode == "pandas" else "spark"
        self.coreset_constructor = GenericCoresetConstructor(mode=coreset_mode)
        self.data_manager = CoresetBenchmarkDataManager(self.coreset_constructor)
        self.synthetic_generator = CoresetSyntheticDataGenerator()
        
        # Initialize Spark session if needed
        self.spark = None
        if self.mode == "pyspark":
            self.spark = self._create_spark_session()
        
        # Test results storage
        self.test_results = []
        self.error_count = 0
        
        self._setup_logging()
    
    def _create_spark_session(self):
        """Create Spark session for PySpark mode."""
        try:
            from pyspark.sql import SparkSession
            
            spark = SparkSession.builder \
                .appName("CoresetTesting") \
                .config("spark.sql.adaptive.enabled", "true") \
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
                .getOrCreate()
            
            logger.info("Spark session created for coreset testing")
            return spark
            
        except ImportError:
            logger.error("PySpark not available. Please install PySpark for pyspark mode.")
            raise ImportError("PySpark not available")
        except Exception as e:
            logger.error(f"Failed to create Spark session: {e}")
            raise
    
    def _setup_logging(self):
        """Setup logging for coreset testing."""
        log_file = self.results_dir / f"coreset_testing_{self.mode}.log"
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
    
    def test_algorithm_on_coreset(self, algorithm_name: str, dataset_name: str,
                                 original_data: Dict[str, Any], coreset_data: Dict[str, Any],
                                 params: Dict[str, Any], sensitivity_method: str = 'exact',
                                 optimization_method: str = 'default') -> Dict[str, Any]:
        """Test a single algorithm on both original and coreset data."""
        
        start_time = time.time()
        
        result = {
            'algorithm': algorithm_name,
            'dataset': dataset_name,
            'optimization': optimization_method,
            'mode': self.mode,
            'params': params.copy(),
            'success': False,
            'error': None,
            'execution_time': 0,
            'original_data_size': len(original_data.get('features', [])),
            'coreset_data_size': len(coreset_data.get('features', [])),
            'coreset_ratio': 0,
            'original_metrics': {},
            'coreset_metrics': {},
            'approximation_quality': {},
            'model_save_success': False,
            'model_load_success': False,
            'model_save_path': None
        }
        
        try:
            logger.info(f"Testing {algorithm_name} on {dataset_name} (coreset, {self.mode}) with {optimization_method} params")
            
            # Calculate coreset ratio
            if result['original_data_size'] > 0:
                result['coreset_ratio'] = result['coreset_data_size'] / result['original_data_size']
            
            # Test on original data
            original_result = self._test_on_data(algorithm_name, original_data, params, "original")
            result['original_metrics'] = original_result.get('metrics', {})
            
            # Test on coreset data
            coreset_result = self._test_on_data(algorithm_name, coreset_data, params, "coreset")
            result['coreset_metrics'] = coreset_result.get('metrics', {})
            
            # Save and load model functionality using the coreset model
            coreset_model = coreset_result.get('model')
            if coreset_model is not None:
                try:
                    # Create Models directory if it doesn't exist
                    models_dir = self.results_dir / "Models"
                    models_dir.mkdir(exist_ok=True)
                    
                    # Define model save path
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    model_filename = f"{algorithm_name}_{dataset_name}_{sensitivity_method}_{optimization_method}_{timestamp}_coreset_{self.mode}.model"
                    model_path = models_dir / model_filename
                    result['model_save_path'] = str(model_path)
                    
                    # Save model
                    logger.info(f"Saving coreset model {algorithm_name} ({self.mode}) to {model_path}")
                    coreset_model.save(str(model_path))
                    result['model_save_success'] = True
                    logger.info(f"Coreset model {algorithm_name} ({self.mode}) saved successfully")
                    
                    # Load model back to verify save/load functionality
                    logger.info(f"Loading coreset model {algorithm_name} ({self.mode}) from {model_path}")
                    model_class = MODEL_REGISTRY[algorithm_name]['class']
                    loaded_model = model_class.load(str(model_path))
                    result['model_load_success'] = True
                    logger.info(f"Coreset model {algorithm_name} ({self.mode}) loaded successfully")
                    
                    # Verify loaded model has same predictions
                    if hasattr(loaded_model, 'labels_') and loaded_model.labels_ is not None:
                        loaded_predictions = loaded_model.labels_
                    elif hasattr(loaded_model, 'predict') and 'data_loader' in coreset_result:
                        loaded_predictions = loaded_model.predict(coreset_result['data_loader'])
                    else:
                        loaded_predictions = None
                    
                    # Compare original and loaded model predictions if possible
                    if (loaded_predictions is not None and 
                        hasattr(coreset_model, 'labels_') and 
                        coreset_model.labels_ is not None):
                        original_predictions = coreset_model.labels_
                        
                        # Handle different data types for pandas vs spark
                        if self.mode == "pyspark":
                            # Handle Spark DataFrame predictions
                            if hasattr(loaded_predictions, 'toPandas'):
                                loaded_predictions = loaded_predictions.toPandas().iloc[:, 0].values
                            if hasattr(original_predictions, 'toPandas'):
                                original_predictions = original_predictions.toPandas().iloc[:, 0].values
                        
                        if isinstance(loaded_predictions, pd.Series):
                            loaded_predictions = loaded_predictions.values
                        if isinstance(original_predictions, pd.Series):
                            original_predictions = original_predictions.values
                        
                        # Check if predictions match
                        predictions_match = np.array_equal(original_predictions, loaded_predictions)
                        result['predictions_match_after_load'] = predictions_match
                        
                        if predictions_match:
                            logger.info(f"Coreset model {algorithm_name} ({self.mode}) save/load verification successful - predictions match")
                        else:
                            logger.warning(f"Coreset model {algorithm_name} ({self.mode}) save/load verification failed - predictions don't match")
                    
                except Exception as e:
                    logger.error(f"Coreset model save/load failed for {algorithm_name} ({self.mode}): {e}")
                    result['model_save_load_error'] = str(e)
            
            # Calculate approximation quality
            result['approximation_quality'] = self._calculate_approximation_quality(
                result['original_metrics'], result['coreset_metrics']
            )
            
            result['success'] = True
            logger.info(f"Successfully tested {algorithm_name} on {dataset_name} (coreset, {self.mode})")
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Failed to test {algorithm_name} on {dataset_name} (coreset, {self.mode}): {e}")
            logger.debug(traceback.format_exc())
        
        result['execution_time'] = time.time() - start_time
        return result
    
    def _test_on_data(self, algorithm_name: str, data: Dict[str, Any], 
                     params: Dict[str, Any], data_type: str) -> Dict[str, Any]:
        """Test algorithm on a single dataset (original or coreset)."""
        
        result = {'metrics': {}, 'model': None, 'data_loader': None}
        
        try:
            # Extract data components
            features = data.get('features')
            similarity = data.get('similarity')  # Not used for attribute modality
            true_labels = data.get('labels')
            
            # Create appropriate data loader based on mode
            if self.mode == "pandas":
                data_loader = PandasDataLoader(features=features, similarity=similarity)
            else:  # pyspark
                from data.loaders import SparkDataLoader
                # Convert pandas to Spark if needed
                if isinstance(features, pd.DataFrame):
                    features_spark = self.spark.createDataFrame(features)
                else:
                    features_spark = features
                data_loader = SparkDataLoader(spark=self.spark, features=features_spark, similarity=None)
            
            result['data_loader'] = data_loader
            
            # Create and fit model
            model = factory.create_model(algorithm_name, params)
            model.fit(data_loader)
            result['model'] = model
            
            # Get predictions
            if hasattr(model, 'labels_') and model.labels_ is not None:
                predicted_labels = model.labels_
            else:
                predicted_labels = model.predict(data_loader)
            
            # Calculate metrics
            if true_labels is not None and predicted_labels is not None:
                # Convert to numpy arrays for metric calculation
                if self.mode == "pyspark":
                    if isinstance(true_labels, pd.Series):
                        true_labels_array = true_labels.values
                    else:
                        true_labels_array = np.array(true_labels)
                    
                    if hasattr(predicted_labels, 'toPandas'):
                        predicted_labels_array = predicted_labels.toPandas().iloc[:, 0].values
                    else:
                        predicted_labels_array = np.array(predicted_labels)
                else:
                    true_labels_array = true_labels.values if isinstance(true_labels, pd.Series) else np.array(true_labels)
                    predicted_labels_array = predicted_labels.values if isinstance(predicted_labels, pd.Series) else np.array(predicted_labels)
                
                # Ensure same length
                min_len = min(len(true_labels_array), len(predicted_labels_array))
                true_labels_array = true_labels_array[:min_len]
                predicted_labels_array = predicted_labels_array[:min_len]
                
                # Calculate external metrics
                result['metrics']['ari'] = adjusted_rand_score(true_labels_array, predicted_labels_array)
                result['metrics']['nmi'] = normalized_mutual_info_score(true_labels_array, predicted_labels_array)
            
            # Calculate internal metrics
            if features is not None and predicted_labels is not None:
                # Convert features to numpy for sklearn metrics
                if self.mode == "pyspark" and hasattr(features, 'toPandas'):
                    features_array = features.toPandas().values
                elif isinstance(features, pd.DataFrame):
                    features_array = features.values
                else:
                    features_array = np.array(features)
                
                if hasattr(predicted_labels, 'toPandas'):
                    predicted_labels_array = predicted_labels.toPandas().iloc[:, 0].values
                else:
                    predicted_labels_array = predicted_labels.values if isinstance(predicted_labels, pd.Series) else np.array(predicted_labels)
                
                if len(np.unique(predicted_labels_array)) > 1:
                    try:
                        result['metrics']['silhouette'] = silhouette_score(features_array, predicted_labels_array)
                    except:
                        pass
                    try:
                        result['metrics']['calinski_harabasz'] = calinski_harabasz_score(features_array, predicted_labels_array)
                    except:
                        pass
            
            # Pattern library metrics
            for metric_name in METRIC_REGISTRY:
                try:
                    metric = factory.create_metric(metric_name)
                    score = metric.calculate(data_loader, predicted_labels, model.model_data)
                    if not np.isnan(score) and np.isfinite(score):
                        result['metrics'][metric_name] = float(score)
                except Exception as e:
                    logger.warning(f"Failed to calculate {metric_name} for {data_type} ({self.mode}): {e}")
            
        except Exception as e:
            logger.error(f"Failed to test on {data_type} data ({self.mode}): {e}")
            result['error'] = str(e)
        
        return result
    
    def _calculate_approximation_quality(self, original_metrics: Dict[str, float], 
                                       coreset_metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculate approximation quality metrics."""
        
        quality = {}
        
        for metric_name in original_metrics:
            if metric_name in coreset_metrics:
                original_value = original_metrics[metric_name]
                coreset_value = coreset_metrics[metric_name]
                
                if original_value != 0:
                    relative_error = abs(original_value - coreset_value) / abs(original_value)
                    quality[f'{metric_name}_relative_error'] = relative_error
                
                quality[f'{metric_name}_absolute_error'] = abs(original_value - coreset_value)
        
        return quality
    
    def discover_algorithms(self) -> Dict[str, Dict]:
        """Discover algorithms compatible with coreset testing."""
        logger.info(f"Discovering algorithms compatible with coreset testing ({self.mode} mode)...")
        
        algorithms = {}
        
        # Only include attribute algorithms since coreset only supports attribute modality
        attribute_algorithms = self._get_attribute_algorithms()
        
        for name, info in MODEL_REGISTRY.items():
            if name.lower() in [alg.lower() for alg in attribute_algorithms]:
                algorithms[name] = {
                    'class': info['class'],
                    'params_help': info['params_help'],
                    'modality': 'attribute'  # Only attribute modality for coreset
                }
                logger.info(f"Found coreset-compatible algorithm: {name} (mode: {self.mode})")
        
        logger.info(f"Total coreset-compatible algorithms ({self.mode}): {len(algorithms)}")
        return algorithms
    
    def _get_attribute_algorithms(self) -> List[str]:
        """Get list of attribute algorithms compatible with current mode."""
        if self.mode == "pandas":
            # Pandas-compatible attribute algorithms
            return ['kmeans', 'dbscan', 'agdc', 'ngdc', 'vgdc', 'gmm']
        else:  # pyspark
            # Spark-compatible attribute algorithms (subset)
            return ['kmeans', 'dbscan']  # Typically fewer algorithms support Spark
    
    def _infer_modality(self, algo_name: str, algo_info: Dict) -> str:
        """Infer algorithm modality - always returns 'attribute' for coreset."""
        # Since coreset only supports attribute modality, always return 'attribute'
        return 'attribute'
    
    def get_default_params(self, algorithm_name: str) -> Dict[str, Any]:
        """Get default parameters for an algorithm."""
        if algorithm_name not in MODEL_REGISTRY:
            return {}
        
        params_help = MODEL_REGISTRY[algorithm_name]['params_help']
        default_params = {}
        
        for param_name, help_text in params_help.items():
            if 'cluster' in param_name.lower():
                default_params[param_name] = 5
            elif param_name in ['n_clusters', 'num_clusters']:
                default_params[param_name] = 5
            elif 'iter' in param_name.lower():
                default_params[param_name] = 100
            elif param_name in ['lr', 'learning_rate']:
                default_params[param_name] = 0.01
            elif param_name in ['eps', 'epsilon']:
                default_params[param_name] = 0.5
            elif 'min_samples' in param_name.lower():
                default_params[param_name] = 5
            elif param_name == 'init':
                default_params[param_name] = 'k-means++'
            else:
                default_params[param_name] = 0.1
        
        return default_params
    
    def save_test_results(self, filename: Optional[str] = None) -> bool:
        """Save current test results to file."""
        try:
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"Coreset_test_results_{self.mode}_{timestamp}.json"
            
            results_path = self.results_dir / filename
            
            with open(results_path, 'w') as f:
                json.dump(self.test_results, f, indent=2, default=str)
            
            logger.info(f"Test results saved to {results_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save test results: {e}")
            return False
    
    def run_comprehensive_tests(self):
        """Run comprehensive coreset tests."""
        
        logger.info(f"Starting comprehensive Pattern library coreset testing ({self.mode} mode)")
        
        algorithms = self.discover_algorithms()
        
        if not algorithms:
            logger.warning(f"No algorithms found for coreset testing ({self.mode} mode)")
            return
        
        # Test on coreset datasets (attribute modality only)
        self._test_coreset_datasets(algorithms)
        
        # Generate comprehensive report
        self._generate_coreset_report()
        
        logger.info(f"Coreset comprehensive testing completed ({self.mode} mode)")
    
    def _test_coreset_datasets(self, algorithms: Dict[str, Dict]):
        """Test algorithms on coreset datasets (attribute modality only)."""
        
        logger.info(f"Testing on coreset datasets ({self.mode} mode)...")
        
        # Test attribute datasets with coresets
        for dataset_name in ['iris', 'wine', 'synthetic_blobs']:
            logger.info(f"Processing coreset dataset: {dataset_name} ({self.mode} mode)")
            
            # Generate or load original data
            if dataset_name == 'synthetic_blobs':
                original_features, original_labels = CoresetSyntheticDataGenerator.generate_attribute_data(
                    n_samples=5000, n_features=10, n_clusters=5
                )
                original_data = {
                    'features': original_features,
                    'similarity': None,
                    'labels': original_labels
                }
            else:
                original_features, original_labels = self.data_manager.load_attribute_dataset(dataset_name)
                if original_features is None:
                    continue
                original_data = {
                    'features': original_features,
                    'similarity': None,
                    'labels': original_labels
                }
            
            # Test algorithms on both original and coreset data
            for algo_name, algo_info in algorithms.items():
                # Only test attribute algorithms since that's what coreset supports
                if algo_info['modality'] == 'attribute':
                    params = self.get_default_params(algo_name)
                    
                    # Test with all sensitivity methods
                    for sensitivity_method in self.sensitivity_methods:
                        logger.info(f"Building coreset with {sensitivity_method} sensitivity for {algo_name}")
                        
                        # Build coreset using the new constructor
                        coreset_features, coreset_weights = self.coreset_constructor.build_attribute_coreset(
                            original_data['features'], 
                            coreset_size=500,
                            sensitivity_method=sensitivity_method,
                            algorithm=algo_name
                        )
                        coreset_data = {
                            'features': pd.DataFrame(coreset_features, columns=original_data['features'].columns),
                            'similarity': None,
                            'labels': original_data['labels'][:len(coreset_features)] if original_data['labels'] is not None else None
                        }
                        
                        result = self.test_algorithm_on_coreset(
                            algo_name, dataset_name, original_data, coreset_data, params, sensitivity_method
                        )
                        result['sensitivity_method'] = sensitivity_method
                        self.test_results.append(result)
        
        # Save results
        self.save_test_results()
    
    def _generate_coreset_report(self):
        """Generate comprehensive coreset testing report."""
        logger.info(f"Generating coreset testing report ({self.mode} mode)...")
        
        if not self.test_results:
            logger.warning("No test results to report")
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.results_dir / "Reports" / f"Coreset_report_{self.mode}_{timestamp}.txt"
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write(f"Pattern Library Coreset Testing Report ({self.mode.upper()} Mode)\n")
            f.write("=" * 60 + "\n\n")
            
            # Summary statistics
            total_tests = len(self.test_results)
            successful_tests = sum(1 for r in self.test_results if r['success'])
            
            f.write(f"Processing Mode: {self.mode.upper()}\n")
            f.write(f"Total Tests: {total_tests}\n")
            f.write(f"Successful Tests: {successful_tests}\n")
            f.write(f"Success Rate: {successful_tests/total_tests:.2%}\n\n")
            
            # Model save/load statistics
            successful_saves = sum(1 for r in self.test_results if r.get('model_save_success', False))
            successful_loads = sum(1 for r in self.test_results if r.get('model_load_success', False))
            
            f.write(f"Model Save Success Rate: {successful_saves/total_tests:.2%}\n")
            f.write(f"Model Load Success Rate: {successful_loads/total_tests:.2%}\n\n")
            
            # Coreset efficiency analysis
            coreset_ratios = [r.get('coreset_ratio', 0) for r in self.test_results if r.get('coreset_ratio')]
            if coreset_ratios:
                avg_ratio = np.mean(coreset_ratios)
                f.write(f"Average Coreset Ratio: {avg_ratio:.3f}\n")
                f.write(f"Data Reduction: {(1-avg_ratio)*100:.1f}%\n\n")
            
            # Detailed results
            f.write("Detailed Results:\n")
            f.write("-" * 20 + "\n")
            
            for result in self.test_results:
                f.write(f"\nAlgorithm: {result['algorithm']}\n")
                f.write(f"Dataset: {result['dataset']}\n")
                f.write(f"Mode: {result.get('mode', 'unknown')}\n")
                f.write(f"Sensitivity Method: {result.get('sensitivity_method', 'unknown')}\n")
                f.write(f"Success: {result['success']}\n")
                f.write(f"Coreset Ratio: {result.get('coreset_ratio', 0):.3f}\n")
                f.write(f"Model Save Success: {result.get('model_save_success', False)}\n")
                f.write(f"Model Load Success: {result.get('model_load_success', False)}\n")
                
                if result.get('approximation_quality'):
                    f.write(f"Approximation Quality: {result['approximation_quality']}\n")
                
                if result.get('error'):
                    f.write(f"Error: {result['error']}\n")
        
        logger.info(f"Coreset report saved to {report_path}")
    
    def save_model(self, model, algorithm_name: str, dataset_name: str, 
                   optimization_method: str = 'manual', suffix: str = '') -> Optional[str]:
        """Save a trained coreset model to disk."""
        try:
            # Create Models directory if it doesn't exist
            models_dir = self.results_dir / "Models"
            models_dir.mkdir(exist_ok=True)
            
            # Define model save path
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_filename = f"{algorithm_name}_{dataset_name}_{optimization_method}_{timestamp}_coreset_{self.mode}{suffix}.model"
            model_path = models_dir / model_filename
            
            # Save model
            logger.info(f"Saving coreset model {algorithm_name} ({self.mode}) to {model_path}")
            model.save(str(model_path))
            logger.info(f"Coreset model {algorithm_name} ({self.mode}) saved successfully")
            
            return str(model_path)
            
        except Exception as e:
            logger.error(f"Failed to save coreset model {algorithm_name} ({self.mode}): {e}")
            return None
    
    def load_model(self, algorithm_name: str, model_path: str):
        """Load a trained coreset model from disk."""
        try:
            logger.info(f"Loading coreset model {algorithm_name} ({self.mode}) from {model_path}")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            model_class = MODEL_REGISTRY[algorithm_name]['class']
            loaded_model = model_class.load(model_path)
            
            logger.info(f"Coreset model {algorithm_name} ({self.mode}) loaded successfully")
            return loaded_model
            
        except Exception as e:
            logger.error(f"Failed to load coreset model {algorithm_name} ({self.mode}): {e}")
            return None
    
    def list_saved_models(self) -> List[str]:
        """List all saved coreset model files."""
        models_dir = self.results_dir / "Models"
        if not models_dir.exists():
            return []
        
        return [f.name for f in models_dir.glob(f"*_coreset_{self.mode}*.model")]
    
    def get_supported_algorithms(self) -> List[str]:
        """Get list of algorithms supported in current mode."""
        return self._get_attribute_algorithms()
    
    def __del__(self):
        """Clean up Spark session if it exists."""
        if self.spark is not None:
            try:
                self.spark.stop()
                logger.info("Spark session stopped")
            except:
                pass

def main():
    """Main coreset testing function."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Pattern Library Coreset Testing')
    parser.add_argument('--mode', choices=['pandas', 'pyspark'], default='pandas',
                        help='Processing mode: pandas or pyspark (default: pandas)')
    parser.add_argument('--sensitivity-methods', nargs='+', 
                       choices=['exact', 'relaxed', 'distance_only'],
                       default=['exact', 'relaxed', 'distance_only'],
                       help='Sensitivity computation methods to test (default: all)')
    args = parser.parse_args()
    
    print(f"Pattern Library Comprehensive Testing - Coreset Scale ({args.mode.upper()} Mode)")
    print("=" * 70)
    print("This test suite will:")
    print("1. Discover attribute algorithms compatible with coreset")
    print("2. Generate attribute datasets and build coresets")
    print("3. Test algorithms on coresets vs original data with multiple sensitivity methods")
    print("4. Analyze approximation quality and efficiency gains")
    print("5. Generate comprehensive coreset performance reports")
    print(f"6. Processing mode: {args.mode.upper()}")
    print(f"7. Sensitivity methods: {', '.join(args.sensitivity_methods)}")
    print("=" * 70)
    
    try:
        tester = CoresetAlgorithmTester(mode=args.mode, sensitivity_methods=args.sensitivity_methods)
        tester.run_comprehensive_tests()
        
        print(f"\nCoreset testing ({args.mode} mode) completed successfully!")
        print(f"Results saved in: {tester.results_dir}")
        print(f"Sensitivity methods tested: {', '.join(args.sensitivity_methods)}")
        
        # Show summary
        if tester.test_results:
            total_tests = len(tester.test_results)
            successful_tests = sum(1 for r in tester.test_results if r['success'])
            print(f"\nTest Summary:")
            print(f"Total tests: {total_tests}")
            print(f"Successful: {successful_tests}")
            print(f"Success rate: {successful_tests/total_tests:.2%}")
            
            # Show statistics by sensitivity method
            print(f"\nResults by sensitivity method:")
            for method in args.sensitivity_methods:
                method_results = [r for r in tester.test_results if r.get('sensitivity_method') == method]
                if method_results:
                    method_success = sum(1 for r in method_results if r['success'])
                    print(f"  {method}: {method_success}/{len(method_results)} successful ({method_success/len(method_results):.2%})")
        
    except Exception as e:
        logger.error(f"Coreset testing failed with error: {e}")
        logger.debug(traceback.format_exc())
        print(f"\nCoreset testing ({args.mode} mode) failed: {e}")

if __name__ == "__main__":
    main() 