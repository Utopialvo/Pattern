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
- Scalable synthetic data generation
- Performance evaluation at scale
- Comprehensive distributed result reporting

Author: Pattern Library Testing Framework
"""

import os
import sys
import json
import logging
import warnings
import traceback
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import time

# Third-party imports
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import requests

# PySpark imports
try:
    from pyspark.sql import SparkSession, DataFrame as SparkDataFrame
    from pyspark.sql.functions import col, rand, when, lit
    from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType
    from pyspark.ml.feature import StandardScaler as SparkStandardScaler, VectorAssembler
    from pyspark.ml.linalg import Vectors, VectorUDT
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
    
    def __init__(self, spark: SparkSession, data_dir: str = "benchmark_data_spark"):
        self.spark = spark
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Large-scale benchmark datasets
        self.benchmark_datasets = {
            'attribute': {
                'sklearn_large': {'samples': 100000, 'features': 20, 'clusters': 5, 'description': 'Large synthetic blobs'},
                'random_large': {'samples': 50000, 'features': 15, 'clusters': 8, 'description': 'Large random dataset'},
                'mixed_gaussian': {'samples': 75000, 'features': 25, 'clusters': 6, 'description': 'Mixed Gaussian clusters'}
            },
            'network': {
                'large_sbm': {'nodes': 10000, 'communities': 20, 'description': 'Large Stochastic Block Model'},
                'scale_free': {'nodes': 15000, 'communities': 15, 'description': 'Large Scale-free network'},
                'small_world': {'nodes': 8000, 'communities': 12, 'description': 'Large Small-world network'}
            },
            'attributed_graph': {
                'large_attr_sbm': {'nodes': 5000, 'features': 30, 'communities': 10, 'description': 'Large attributed SBM'},
                'complex_attr_graph': {'nodes': 7500, 'features': 40, 'communities': 12, 'description': 'Complex attributed graph'}
            }
        }
        
        # Benchmark performance expectations
        self.benchmark_performance = {
            'sklearn_large': {'silhouette_target': 0.4, 'time_limit': 300},
            'large_sbm': {'modularity_target': 0.3, 'time_limit': 600},
            'large_attr_sbm': {'combined_metric_target': 0.35, 'time_limit': 900}
        }
    
    def create_large_attribute_dataset(self, name: str) -> Tuple[SparkDataFrame, SparkDataFrame]:
        """Create large-scale attribute dataset using Spark."""
        
        dataset_config = self.benchmark_datasets['attribute'][name]
        
        if name == 'sklearn_large':
            # Generate large sklearn-style dataset
            n_samples = dataset_config['samples']
            n_features = dataset_config['features']
            n_clusters = dataset_config['clusters']
            
            # Use sklearn for generation, then convert to Spark
            X, y = make_blobs(n_samples=n_samples, centers=n_clusters, 
                             n_features=n_features, cluster_std=1.5, random_state=42)
            
            # Create Spark DataFrame
            feature_columns = [f'feature_{i}' for i in range(n_features)]
            data_list = [(float(y[i]),) + tuple(float(x) for x in X[i]) for i in range(len(X))]
            
            schema = StructType([StructField('true_label', DoubleType(), True)] + 
                               [StructField(col, DoubleType(), True) for col in feature_columns])
            
            df = self.spark.createDataFrame(data_list, schema)
            
            # Split features and labels
            features_df = df.select(*feature_columns)
            labels_df = df.select('true_label')
            
            return features_df, labels_df
            
        elif name == 'random_large':
            # Generate large random dataset with artificial clusters
            n_samples = dataset_config['samples']
            n_features = dataset_config['features']
            n_clusters = dataset_config['clusters']
            
            # Create random data with cluster structure
            cluster_centers = np.random.randn(n_clusters, n_features) * 5
            
            data_list = []
            for i in range(n_samples):
                cluster_id = np.random.randint(0, n_clusters)
                point = cluster_centers[cluster_id] + np.random.randn(n_features) * 2
                data_list.append((float(cluster_id),) + tuple(float(x) for x in point))
            
            feature_columns = [f'feature_{i}' for i in range(n_features)]
            schema = StructType([StructField('true_label', DoubleType(), True)] + 
                               [StructField(col, DoubleType(), True) for col in feature_columns])
            
            df = self.spark.createDataFrame(data_list, schema)
            features_df = df.select(*feature_columns)
            labels_df = df.select('true_label')
            
            return features_df, labels_df
        
        return None, None
    
    def create_large_network_dataset(self, name: str) -> Tuple[None, SparkDataFrame, SparkDataFrame]:
        """Create large-scale network dataset using Spark."""
        
        dataset_config = self.benchmark_datasets['network'][name]
        
        if name == 'large_sbm':
            n_nodes = dataset_config['nodes']
            n_communities = dataset_config['communities']
            p_in = 0.1
            p_out = 0.01
            
            # Generate SBM with NetworkX (for structure) then convert to Spark
            community_sizes = [n_nodes // n_communities] * n_communities
            community_sizes[-1] += n_nodes % n_communities
            
            logger.info(f"Generating large SBM with {n_nodes} nodes and {n_communities} communities")
            
            # Create adjacency matrix data
            edges = []
            node_communities = []
            
            # Assign nodes to communities
            node_id = 0
            for comm_id, size in enumerate(community_sizes):
                for _ in range(size):
                    node_communities.append(comm_id)
                    node_id += 1
            
            # Generate edges based on SBM probabilities
            for i in range(n_nodes):
                for j in range(i + 1, n_nodes):
                    if node_communities[i] == node_communities[j]:
                        prob = p_in
                    else:
                        prob = p_out
                    
                    if np.random.random() < prob:
                        edges.append((i, j, 1.0))
            
            # Create Spark DataFrame for adjacency matrix (edge list format)
            edge_schema = StructType([
                StructField('src', IntegerType(), True),
                StructField('dst', IntegerType(), True),
                StructField('weight', DoubleType(), True)
            ])
            
            edges_df = self.spark.createDataFrame(edges, edge_schema)
            
            # Create labels DataFrame
            labels_data = [(i, float(node_communities[i])) for i in range(n_nodes)]
            labels_schema = StructType([
                StructField('node_id', IntegerType(), True),
                StructField('true_label', DoubleType(), True)
            ])
            
            labels_df = self.spark.createDataFrame(labels_data, labels_schema)
            
            logger.info(f"Generated network with {edges_df.count()} edges")
            
            return None, edges_df, labels_df
        
        return None, None, None
    
    def create_large_attributed_graph_dataset(self, name: str) -> Tuple[SparkDataFrame, SparkDataFrame, SparkDataFrame]:
        """Create large-scale attributed graph dataset using Spark."""
        
        dataset_config = self.benchmark_datasets['attributed_graph'][name]
        
        if name == 'large_attr_sbm':
            n_nodes = dataset_config['nodes']
            n_features = dataset_config['features']
            n_communities = dataset_config['communities']
            
            logger.info(f"Generating large attributed graph with {n_nodes} nodes, {n_features} features, {n_communities} communities")
            
            # First generate network structure
            _, edges_df, labels_df = self.create_large_network_dataset('large_sbm')
            
            # Generate node features correlated with communities
            # Get community assignments
            community_assignments = labels_df.collect()
            community_dict = {row['node_id']: int(row['true_label']) for row in community_assignments}
            
            # Generate features for each community
            community_centers = np.random.randn(n_communities, n_features) * 3
            
            features_data = []
            for node_id in range(n_nodes):
                community = community_dict[node_id]
                # Generate features centered around community center
                features = community_centers[community] + np.random.randn(n_features) * 1.5
                features_data.append((node_id,) + tuple(float(f) for f in features))
            
            # Create features DataFrame
            feature_columns = [f'feature_{i}' for i in range(n_features)]
            features_schema = StructType([StructField('node_id', IntegerType(), True)] + 
                                       [StructField(col, DoubleType(), True) for col in feature_columns])
            
            features_df = self.spark.createDataFrame(features_data, features_schema)
            
            return features_df, edges_df, labels_df
        
        return None, None, None

class SparkSyntheticDataGenerator:
    """Generates large-scale synthetic datasets using PySpark."""
    
    def __init__(self, spark: SparkSession):
        self.spark = spark
    
    def generate_large_attribute_data(self, n_samples: int = 50000, n_features: int = 20, 
                                     n_clusters: int = 5, scenario: str = 'blobs') -> Tuple[SparkDataFrame, SparkDataFrame]:
        """Generate large-scale synthetic attribute data using Spark."""
        
        logger.info(f"Generating large attribute dataset: {n_samples} samples, {n_features} features, {n_clusters} clusters")
        
        if scenario == 'blobs':
            # Generate cluster centers
            cluster_centers = np.random.randn(n_clusters, n_features) * 5
            
            # Generate data points
            data_list = []
            for i in range(n_samples):
                cluster_id = np.random.randint(0, n_clusters)
                point = cluster_centers[cluster_id] + np.random.randn(n_features) * 2
                data_list.append((float(cluster_id),) + tuple(float(x) for x in point))
            
            feature_columns = [f'feature_{i}' for i in range(n_features)]
            schema = StructType([StructField('true_label', DoubleType(), True)] + 
                               [StructField(col, DoubleType(), True) for col in feature_columns])
            
            df = self.spark.createDataFrame(data_list, schema)
            
            # Normalize features using Spark ML
            assembler = VectorAssembler(inputCols=feature_columns, outputCol="features_vector")
            df_vector = assembler.transform(df)
            
            scaler = SparkStandardScaler(inputCol="features_vector", outputCol="scaled_features", withStd=True, withMean=True)
            scaler_model = scaler.fit(df_vector)
            df_scaled = scaler_model.transform(df_vector)
            
            # Split back into individual columns (simplified approach)
            features_df = df.select(*feature_columns)
            labels_df = df.select('true_label')
            
            return features_df, labels_df
        
        elif scenario == 'sparse_clusters':
            # Generate sparse cluster scenario
            cluster_centers = np.random.randn(n_clusters, n_features) * 10
            
            data_list = []
            for i in range(n_samples):
                cluster_id = np.random.randint(0, n_clusters)
                # Make clusters more separated
                point = cluster_centers[cluster_id] + np.random.randn(n_features) * 1.0
                data_list.append((float(cluster_id),) + tuple(float(x) for x in point))
            
            feature_columns = [f'feature_{i}' for i in range(n_features)]
            schema = StructType([StructField('true_label', DoubleType(), True)] + 
                               [StructField(col, DoubleType(), True) for col in feature_columns])
            
            df = self.spark.createDataFrame(data_list, schema)
            features_df = df.select(*feature_columns)
            labels_df = df.select('true_label')
            
            return features_df, labels_df
        
        return None, None
    
    def generate_large_network_data(self, n_nodes: int = 10000, n_communities: int = 10,
                                   p_in: float = 0.1, p_out: float = 0.01) -> Tuple[None, SparkDataFrame, SparkDataFrame]:
        """Generate large-scale synthetic network data using Spark."""
        
        logger.info(f"Generating large network: {n_nodes} nodes, {n_communities} communities")
        
        # Assign nodes to communities
        community_sizes = [n_nodes // n_communities] * n_communities
        community_sizes[-1] += n_nodes % n_communities
        
        node_communities = []
        node_id = 0
        for comm_id, size in enumerate(community_sizes):
            for _ in range(size):
                node_communities.append(comm_id)
                node_id += 1
        
        # Generate edges efficiently (sample approach for large graphs)
        edges = []
        max_edges = min(100000, n_nodes * 10)  # Limit edges for memory efficiency
        
        for _ in range(max_edges):
            i = np.random.randint(0, n_nodes)
            j = np.random.randint(0, n_nodes)
            
            if i != j:
                if node_communities[i] == node_communities[j]:
                    prob = p_in
                else:
                    prob = p_out
                
                if np.random.random() < prob:
                    edges.append((i, j, 1.0))
        
        # Remove duplicates
        edges = list(set(edges))
        
        # Create Spark DataFrames
        edge_schema = StructType([
            StructField('src', IntegerType(), True),
            StructField('dst', IntegerType(), True),
            StructField('weight', DoubleType(), True)
        ])
        
        edges_df = self.spark.createDataFrame(edges, edge_schema)
        
        labels_data = [(i, float(node_communities[i])) for i in range(n_nodes)]
        labels_schema = StructType([
            StructField('node_id', IntegerType(), True),
            StructField('true_label', DoubleType(), True)
        ])
        
        labels_df = self.spark.createDataFrame(labels_data, labels_schema)
        
        logger.info(f"Generated network with {len(edges)} edges")
        
        return None, edges_df, labels_df

class SparkAlgorithmTester:
    """Tests Pattern library algorithms at PySpark scale."""
    
    def __init__(self, results_dir: str = "test_results_spark"):
        if not SPARK_AVAILABLE:
            raise ImportError("PySpark is required for distributed testing")
        
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        self.spark = self._create_spark_session()
        self.data_manager = SparkBenchmarkDataManager(self.spark)
        self.synthetic_generator = SparkSyntheticDataGenerator(self.spark)
        self.test_results = []
        
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
        log_file = self.results_dir / f"spark_test_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
    
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
        elif any(keyword in name_lower for keyword in ['dmon', 'gnn', 'graph', 'node2vec']):
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
            'spark_partitions': 0
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