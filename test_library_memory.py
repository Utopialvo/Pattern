#!/usr/bin/env python3
"""
Test Library for Pattern - In-Memory Scale
===========================================

This module provides comprehensive testing for the Pattern library at in-memory scale.
It automatically discovers implemented algorithms, downloads benchmark datasets,
generates synthetic data, and evaluates performance using both default hyperparameters
and Optuna optimization.

Features:
- Automatic algorithm and metric discovery
- Benchmark dataset downloading for all modalities
- Synthetic data generation for each modality
- Performance evaluation with default and optimized hyperparameters
- Comprehensive result reporting and analysis

Author: Pattern Library Testing Framework
"""

import os
import sys
import json
import logging
import warnings
import importlib
import traceback
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime
import time

# Third-party imports
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
import requests
import zipfile
import tarfile
from urllib.parse import urlparse

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

class BenchmarkDataManager:
    """Manages benchmark dataset downloading and preprocessing for all modalities."""
    
    def __init__(self, data_dir: str = "benchmark_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Benchmark datasets by modality
        self.benchmark_datasets = {
            'attribute': {
                'iris': {
                    'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                    'description': 'Classic iris flower dataset',
                    'expected_clusters': 3
                },
                'wine': {
                    'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',
                    'description': 'Wine recognition dataset',
                    'expected_clusters': 3
                },
                'breast_cancer': {
                    'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data',
                    'description': 'Breast cancer Wisconsin dataset',
                    'expected_clusters': 2
                },
                'seeds': {
                    'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt',
                    'description': 'Seeds dataset',
                    'expected_clusters': 3
                }
            },
            'network': {
                'karate': {
                    'description': 'Zachary karate club network',
                    'expected_clusters': 2,
                    'builtin': True
                },
                'dolphins': {
                    'url': 'http://www-personal.umich.edu/~mejn/netdata/dolphins.zip',
                    'description': 'Dolphin social network',
                    'expected_clusters': 2
                },
                'football': {
                    'url': 'http://www-personal.umich.edu/~mejn/netdata/football.zip',
                    'description': 'American college football network',
                    'expected_clusters': 12
                },
                'polbooks': {
                    'url': 'http://www-personal.umich.edu/~mejn/netdata/polbooks.zip',
                    'description': 'Political books co-purchasing network',
                    'expected_clusters': 3
                }
            },
            'attributed_graph': {
                'cora': {
                    'url': 'https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz',
                    'description': 'Cora citation network with features',
                    'expected_clusters': 7
                },
                'citeseer': {
                    'url': 'https://linqs-data.soe.ucsc.edu/public/lbc/citeseer.tgz',
                    'description': 'CiteSeer citation network with features',
                    'expected_clusters': 6
                },
                'pubmed': {
                    'url': 'https://linqs-data.soe.ucsc.edu/public/Pubmed-Diabetes.tgz',
                    'description': 'PubMed diabetes citation network',
                    'expected_clusters': 3
                }
            }
        }
        
        # Benchmark performance values from literature
        self.benchmark_performance = {
            'iris': {'silhouette': 0.55, 'calinski_harabasz': 561.6},
            'wine': {'silhouette': 0.27, 'calinski_harabasz': 561.9},
            'karate': {'modularity': 0.37, 'anui': 0.65},
            'dolphins': {'modularity': 0.52, 'anui': 0.71},
            'cora': {'modularity': 0.74, 'silhouette': 0.42}
        }
        
    def download_file(self, url: str, filename: str) -> bool:
        """Download a file from URL."""
        try:
            filepath = self.data_dir / filename
            if filepath.exists():
                logger.info(f"File {filename} already exists, skipping download")
                return True
                
            logger.info(f"Downloading {filename} from {url}")
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Extract if archive
            if filename.endswith(('.zip', '.tgz', '.tar.gz')):
                self._extract_archive(filepath)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")
            return False
    
    def _extract_archive(self, filepath: Path):
        """Extract archive files."""
        try:
            if filepath.suffix == '.zip':
                with zipfile.ZipFile(filepath, 'r') as zip_ref:
                    zip_ref.extractall(filepath.parent)
            elif filepath.suffix in ['.tgz', '.gz']:
                with tarfile.open(filepath, 'r:gz') as tar_ref:
                    tar_ref.extractall(filepath.parent)
        except Exception as e:
            logger.error(f"Failed to extract {filepath}: {e}")
    
    def load_attribute_dataset(self, name: str) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Load attribute-based dataset."""
        dataset_info = self.benchmark_datasets['attribute'][name]
        
        if name == 'iris':
            if not self.download_file(dataset_info['url'], 'iris.data'):
                return None, None
            
            columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
            df = pd.read_csv(self.data_dir / 'iris.data', names=columns)
            features = df.drop('class', axis=1)
            labels = pd.Categorical(df['class']).codes
            return features, pd.Series(labels, name='true_labels')
            
        elif name == 'wine':
            if not self.download_file(dataset_info['url'], 'wine.data'):
                return None, None
            
            df = pd.read_csv(self.data_dir / 'wine.data', header=None)
            features = df.iloc[:, 1:]
            labels = df.iloc[:, 0] - 1  # Convert to 0-based
            return features, pd.Series(labels, name='true_labels')
            
        elif name == 'breast_cancer':
            if not self.download_file(dataset_info['url'], 'wdbc.data'):
                return None, None
            
            df = pd.read_csv(self.data_dir / 'wdbc.data', header=None)
            features = df.iloc[:, 2:]  # Skip ID and diagnosis
            labels = pd.Categorical(df.iloc[:, 1]).codes
            return features, pd.Series(labels, name='true_labels')
            
        elif name == 'seeds':
            if not self.download_file(dataset_info['url'], 'seeds_dataset.txt'):
                return None, None
            
            df = pd.read_csv(self.data_dir / 'seeds_dataset.txt', sep='\t', header=None)
            features = df.iloc[:, :-1]
            labels = df.iloc[:, -1] - 1  # Convert to 0-based
            return features, pd.Series(labels, name='true_labels')
        
        return None, None
    
    def load_network_dataset(self, name: str) -> Tuple[Optional[pd.DataFrame], pd.DataFrame]:
        """Load network dataset."""
        dataset_info = self.benchmark_datasets['network'][name]
        
        if name == 'karate':
            G = nx.karate_club_graph()
            adj_matrix = pd.DataFrame(nx.adjacency_matrix(G).toarray())
            # Ground truth communities
            true_labels = [0 if G.nodes[n]['club'] == 'Mr. Hi' else 1 for n in G.nodes()]
            return None, adj_matrix
            
        elif name == 'dolphins':
            if not self.download_file(dataset_info['url'], 'dolphins.zip'):
                return None, None
            
            # Parse GML file after extraction
            gml_path = self.data_dir / 'dolphins.gml'
            if gml_path.exists():
                G = nx.read_gml(gml_path)
                adj_matrix = pd.DataFrame(nx.adjacency_matrix(G).toarray())
                return None, adj_matrix
        
        # Add more network datasets as needed
        return None, None
    
    def load_attributed_graph_dataset(self, name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load attributed graph dataset."""
        dataset_info = self.benchmark_datasets['attributed_graph'][name]
        
        if name == 'cora':
            # Check if local cora.npz exists
            cora_path = Path('cora.npz')
            if cora_path.exists():
                data = np.load(cora_path, allow_pickle=True)
                features = pd.DataFrame(data['features'])
                adj_matrix = pd.DataFrame(data['adj_matrix'])
                return features, adj_matrix
            
            # Download and process
            if not self.download_file(dataset_info['url'], 'cora.tgz'):
                return None, None
            
            # Process cora dataset files
            # This would need specific parsing logic for the Cora format
            
        return None, None

class SyntheticDataGenerator:
    """Generates synthetic datasets for each modality."""
    
    @staticmethod
    def generate_attribute_data(n_samples: int = 1000, n_features: int = 10, 
                               n_clusters: int = 3, cluster_std: float = 1.0,
                               scenario: str = 'blobs') -> Tuple[pd.DataFrame, pd.Series]:
        """Generate synthetic attribute data."""
        
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
    def generate_network_data(n_nodes: int = 100, n_communities: int = 3,
                             p_in: float = 0.3, p_out: float = 0.05,
                             scenario: str = 'sbm') -> Tuple[None, pd.DataFrame, pd.Series]:
        """Generate synthetic network data."""
        
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
    def generate_attributed_graph_data(n_nodes: int = 500, n_features: int = 20,
                                      n_communities: int = 3, p_in: float = 0.3,
                                      p_out: float = 0.05) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """Generate synthetic attributed graph data."""
        
        # Generate network structure
        _, adj_matrix, true_labels = SyntheticDataGenerator.generate_network_data(
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

class AlgorithmTester:
    """Tests Pattern library algorithms with various configurations."""
    
    def __init__(self, results_dir: str = "test_results_memory"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.data_manager = BenchmarkDataManager()
        self.synthetic_generator = SyntheticDataGenerator()
        
        # Test results storage
        self.test_results = []
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = self.results_dir / f"test_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    def discover_algorithms(self) -> Dict[str, Dict]:
        """Discover all implemented algorithms."""
        logger.info("Discovering implemented algorithms...")
        
        algorithms = {}
        for name, info in MODEL_REGISTRY.items():
            algorithms[name] = {
                'class': info['class'],
                'params_help': info['params_help'],
                'modality': self._infer_modality(name, info)
            }
            logger.info(f"Found algorithm: {name} (modality: {algorithms[name]['modality']})")
        
        logger.info(f"Total algorithms discovered: {len(algorithms)}")
        return algorithms
    
    def discover_metrics(self) -> Dict[str, Any]:
        """Discover all implemented metrics."""
        logger.info("Discovering implemented metrics...")
        
        metrics = {}
        for name, metric_class in METRIC_REGISTRY.items():
            metrics[name] = metric_class
            logger.info(f"Found metric: {name}")
        
        logger.info(f"Total metrics discovered: {len(metrics)}")
        return metrics
    
    def _infer_modality(self, algo_name: str, algo_info: Dict) -> str:
        """Infer the modality of an algorithm based on its name and parameters."""
        name_lower = algo_name.lower()
        
        # Check for network-specific algorithms
        if any(keyword in name_lower for keyword in ['spectral', 'louvain', 'modularity']):
            return 'network'
        
        # Check for attributed graph algorithms
        if any(keyword in name_lower for keyword in ['dmon', 'gnn', 'graph', 'node2vec']):
            return 'attributed_graph'
        
        # Default to attribute-based
        return 'attribute'
    
    def get_default_params(self, algorithm_name: str) -> Dict[str, Any]:
        """Get default parameters for an algorithm."""
        if algorithm_name not in MODEL_REGISTRY:
            return {}
        
        params_help = MODEL_REGISTRY[algorithm_name]['params_help']
        default_params = {}
        
        # Define sensible defaults based on parameter names
        for param_name, description in params_help.items():
            desc_lower = description.lower()
            
            if 'cluster' in param_name.lower() and 'number' in desc_lower:
                default_params[param_name] = 3
            elif param_name.lower() in ['eps', 'epsilon']:
                default_params[param_name] = 0.5
            elif 'min_samples' in param_name.lower():
                default_params[param_name] = 5
            elif 'init' in param_name.lower():
                default_params[param_name] = 'k-means++'
            elif 'max_iter' in param_name.lower():
                default_params[param_name] = 300
            elif 'resolution' in param_name.lower():
                default_params[param_name] = 1.0
            elif 'lr' in param_name.lower() or 'learning_rate' in param_name.lower():
                default_params[param_name] = 0.01
            elif 'epoch' in param_name.lower():
                default_params[param_name] = 100
            elif 'hidden' in param_name.lower() and 'dim' in param_name.lower():
                default_params[param_name] = 64
            elif 'dropout' in param_name.lower():
                default_params[param_name] = 0.1
            
        return default_params
    
    def test_algorithm_on_dataset(self, algorithm_name: str, dataset_name: str,
                                 features: pd.DataFrame, similarity: Optional[pd.DataFrame],
                                 true_labels: Optional[pd.Series], params: Dict[str, Any],
                                 optimization_method: str = 'default') -> Dict[str, Any]:
        """Test a single algorithm on a dataset."""
        
        start_time = time.time()
        result = {
            'algorithm': algorithm_name,
            'dataset': dataset_name,
            'optimization': optimization_method,
            'params': params.copy(),
            'success': False,
            'error': None,
            'execution_time': 0,
            'metrics': {}
        }
        
        try:
            logger.info(f"Testing {algorithm_name} on {dataset_name} with {optimization_method} params")
            
            # Create data loader
            data_loader = PandasDataLoader(features=features, similarity=similarity)
            
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
                # External metrics (require ground truth)
                result['metrics']['ari'] = adjusted_rand_score(true_labels, predicted_labels)
                result['metrics']['nmi'] = normalized_mutual_info_score(true_labels, predicted_labels)
            
            # Internal metrics (using Pattern library metrics)
            for metric_name in METRIC_REGISTRY:
                try:
                    metric = factory.create_metric(metric_name)
                    score = metric.calculate(data_loader, predicted_labels, model.model_data)
                    if not np.isnan(score):
                        result['metrics'][metric_name] = score
                except Exception as e:
                    logger.warning(f"Failed to calculate {metric_name}: {e}")
            
            result['success'] = True
            logger.info(f"Successfully tested {algorithm_name} on {dataset_name}")
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Failed to test {algorithm_name} on {dataset_name}: {e}")
            logger.debug(traceback.format_exc())
        
        result['execution_time'] = time.time() - start_time
        return result
    
    def optimize_hyperparameters(self, algorithm_name: str, dataset_name: str,
                                features: pd.DataFrame, similarity: Optional[pd.DataFrame],
                                true_labels: Optional[pd.Series], n_trials: int = 20) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna."""
        
        logger.info(f"Optimizing hyperparameters for {algorithm_name} on {dataset_name}")
        
        try:
            # Create data loader
            data_loader = PandasDataLoader(features=features, similarity=similarity)
            
            # Get parameter grid for optimization
            param_grid = self._get_param_grid(algorithm_name)
            
            if not param_grid:
                logger.warning(f"No parameter grid defined for {algorithm_name}")
                return self.get_default_params(algorithm_name)
            
            # Create optimizer
            optimizer = TPESearch(n_trials=min(n_trials, 50))  # Limit trials for memory testing
            
            # Determine appropriate metric
            metric_name = self._get_optimization_metric(algorithm_name)
            metric = factory.create_metric(metric_name) if metric_name else None
            
            if metric is None:
                logger.warning(f"No metric available for optimization of {algorithm_name}")
                return self.get_default_params(algorithm_name)
            
            # Run optimization
            model_class = MODEL_REGISTRY[algorithm_name]['class']
            best_params = optimizer.find_best(
                model_class=model_class,
                data_loader=data_loader,
                param_grid=param_grid,
                metric=metric
            )
            
            logger.info(f"Optimization completed for {algorithm_name}: {best_params}")
            return best_params
            
        except Exception as e:
            logger.error(f"Hyperparameter optimization failed for {algorithm_name}: {e}")
            return self.get_default_params(algorithm_name)
    
    def _get_param_grid(self, algorithm_name: str) -> Dict[str, List[Any]]:
        """Get parameter grid for hyperparameter optimization."""
        
        # Define parameter grids for different algorithms
        param_grids = {
            'kmeans': {
                'n_clusters': [2, 3, 4, 5, 6],
                'init': ['k-means++', 'random'],
                'max_iter': [100, 200, 300]
            },
            'dbscan': {
                'eps': [0.1, 0.3, 0.5, 0.7, 1.0],
                'min_samples': [3, 5, 10, 15]
            },
            'spectral': {
                'n_clusters': [2, 3, 4, 5, 6],
                'assign_labels': ['kmeans', 'discretize']
            },
            'louvain': {
                'resolution': [0.5, 1.0, 1.5, 2.0]
            }
        }
        
        return param_grids.get(algorithm_name, {})
    
    def _get_optimization_metric(self, algorithm_name: str) -> str:
        """Get appropriate metric for optimization."""
        
        # Map algorithms to their appropriate metrics
        metric_mapping = {
            'kmeans': 'attribute',
            'dbscan': 'attribute',
            'spectral': 'graph',
            'louvain': 'graph',
            'dmon': 'attribute-graph'
        }
        
        return metric_mapping.get(algorithm_name, 'attribute')
    
    def run_comprehensive_tests(self):
        """Run comprehensive tests on all algorithms and datasets."""
        
        logger.info("Starting comprehensive Pattern library testing (Memory Scale)")
        
        # Discover algorithms and metrics
        algorithms = self.discover_algorithms()
        metrics = self.discover_metrics()
        
        # Test on benchmark datasets
        self._test_benchmark_datasets(algorithms)
        
        # Test on synthetic datasets
        self._test_synthetic_datasets(algorithms)
        
        # Generate comprehensive report
        self._generate_report()
        
        logger.info("Comprehensive testing completed")
    
    def _test_benchmark_datasets(self, algorithms: Dict[str, Dict]):
        """Test algorithms on benchmark datasets."""
        
        logger.info("Testing on benchmark datasets...")
        
        # Test attribute datasets
        for dataset_name in self.data_manager.benchmark_datasets['attribute']:
            logger.info(f"Loading benchmark dataset: {dataset_name}")
            
            features, true_labels = self.data_manager.load_attribute_dataset(dataset_name)
            if features is None:
                logger.warning(f"Failed to load {dataset_name}")
                continue
            
            # Test relevant algorithms
            for algo_name, algo_info in algorithms.items():
                if algo_info['modality'] == 'attribute':
                    
                    # Test with default parameters
                    default_params = self.get_default_params(algo_name)
                    result = self.test_algorithm_on_dataset(
                        algo_name, dataset_name, features, None, true_labels,
                        default_params, 'default'
                    )
                    self.test_results.append(result)
                    
                    # Test with optimized parameters
                    optimized_params = self.optimize_hyperparameters(
                        algo_name, dataset_name, features, None, true_labels
                    )
                    result = self.test_algorithm_on_dataset(
                        algo_name, dataset_name, features, None, true_labels,
                        optimized_params, 'optimized'
                    )
                    self.test_results.append(result)
        
        # Test network datasets
        for dataset_name in self.data_manager.benchmark_datasets['network']:
            if dataset_name == 'karate':  # Test only Karate club for memory tests
                logger.info(f"Loading benchmark dataset: {dataset_name}")
                
                features, adj_matrix = self.data_manager.load_network_dataset(dataset_name)
                if adj_matrix is None:
                    continue
                
                # Create ground truth labels for karate club
                G = nx.karate_club_graph()
                true_labels = pd.Series([0 if G.nodes[n]['club'] == 'Mr. Hi' else 1 for n in G.nodes()])
                
                # Test relevant algorithms
                for algo_name, algo_info in algorithms.items():
                    if algo_info['modality'] == 'network':
                        
                        # Test with default parameters
                        default_params = self.get_default_params(algo_name)
                        result = self.test_algorithm_on_dataset(
                            algo_name, dataset_name, features, adj_matrix, true_labels,
                            default_params, 'default'
                        )
                        self.test_results.append(result)
                        
                        # Test with optimized parameters
                        optimized_params = self.optimize_hyperparameters(
                            algo_name, dataset_name, features, adj_matrix, true_labels
                        )
                        result = self.test_algorithm_on_dataset(
                            algo_name, dataset_name, features, adj_matrix, true_labels,
                            optimized_params, 'optimized'
                        )
                        self.test_results.append(result)
    
    def _test_synthetic_datasets(self, algorithms: Dict[str, Dict]):
        """Test algorithms on synthetic datasets."""
        
        logger.info("Testing on synthetic datasets...")
        
        # Synthetic attribute data scenarios
        attribute_scenarios = [
            {'name': 'blobs_easy', 'params': {'n_samples': 500, 'n_features': 5, 'n_clusters': 3, 'cluster_std': 0.8}},
            {'name': 'blobs_hard', 'params': {'n_samples': 500, 'n_features': 10, 'n_clusters': 5, 'cluster_std': 2.0}},
            {'name': 'circles', 'params': {'n_samples': 500, 'scenario': 'circles'}},
            {'name': 'moons', 'params': {'n_samples': 500, 'scenario': 'moons'}}
        ]
        
        for scenario in attribute_scenarios:
            logger.info(f"Generating synthetic dataset: {scenario['name']}")
            
            features, true_labels = self.synthetic_generator.generate_attribute_data(**scenario['params'])
            
            # Test relevant algorithms
            for algo_name, algo_info in algorithms.items():
                if algo_info['modality'] == 'attribute':
                    
                    # Test with default parameters
                    default_params = self.get_default_params(algo_name)
                    # Adjust n_clusters for scenarios
                    if 'n_clusters' in default_params and scenario['name'].startswith('blobs'):
                        default_params['n_clusters'] = scenario['params'].get('n_clusters', 3)
                    
                    result = self.test_algorithm_on_dataset(
                        algo_name, f"synthetic_{scenario['name']}", features, None, true_labels,
                        default_params, 'default'
                    )
                    self.test_results.append(result)
        
        # Synthetic network data scenarios
        network_scenarios = [
            {'name': 'sbm_small', 'params': {'n_nodes': 100, 'n_communities': 3, 'p_in': 0.4, 'p_out': 0.05}},
            {'name': 'sbm_medium', 'params': {'n_nodes': 200, 'n_communities': 4, 'p_in': 0.3, 'p_out': 0.02}},
        ]
        
        for scenario in network_scenarios:
            logger.info(f"Generating synthetic network: {scenario['name']}")
            
            _, adj_matrix, true_labels = self.synthetic_generator.generate_network_data(**scenario['params'])
            
            # Test relevant algorithms
            for algo_name, algo_info in algorithms.items():
                if algo_info['modality'] == 'network':
                    
                    default_params = self.get_default_params(algo_name)
                    if 'n_clusters' in default_params:
                        default_params['n_clusters'] = scenario['params']['n_communities']
                    
                    result = self.test_algorithm_on_dataset(
                        algo_name, f"synthetic_{scenario['name']}", None, adj_matrix, true_labels,
                        default_params, 'default'
                    )
                    self.test_results.append(result)
        
        # Synthetic attributed graph scenarios
        ag_scenarios = [
            {'name': 'attr_graph_small', 'params': {'n_nodes': 200, 'n_features': 10, 'n_communities': 3}},
            {'name': 'attr_graph_medium', 'params': {'n_nodes': 300, 'n_features': 15, 'n_communities': 4}},
        ]
        
        for scenario in ag_scenarios:
            logger.info(f"Generating synthetic attributed graph: {scenario['name']}")
            
            features, adj_matrix, true_labels = self.synthetic_generator.generate_attributed_graph_data(**scenario['params'])
            
            # Test relevant algorithms
            for algo_name, algo_info in algorithms.items():
                if algo_info['modality'] == 'attributed_graph':
                    
                    default_params = self.get_default_params(algo_name)
                    if 'num_clusters' in default_params:
                        default_params['num_clusters'] = scenario['params']['n_communities']
                    
                    result = self.test_algorithm_on_dataset(
                        algo_name, f"synthetic_{scenario['name']}", features, adj_matrix, true_labels,
                        default_params, 'default'
                    )
                    self.test_results.append(result)
    
    def _generate_report(self):
        """Generate comprehensive test report."""
        
        logger.info("Generating comprehensive test report...")
        
        # Convert results to DataFrame for analysis
        df_results = pd.DataFrame(self.test_results)
        
        # Save detailed results
        results_file = self.results_dir / f"detailed_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df_results.to_csv(results_file, index=False)
        
        # Generate summary report
        summary = self._create_summary_report(df_results)
        
        summary_file = self.results_dir / f"summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        logger.info("=" * 80)
        logger.info("PATTERN LIBRARY TEST SUMMARY (MEMORY SCALE)")
        logger.info("=" * 80)
        logger.info(f"Total tests executed: {len(self.test_results)}")
        logger.info(f"Successful tests: {sum(1 for r in self.test_results if r['success'])}")
        logger.info(f"Failed tests: {sum(1 for r in self.test_results if not r['success'])}")
        logger.info(f"Average execution time: {np.mean([r['execution_time'] for r in self.test_results]):.2f} seconds")
        
        # Best performing algorithms
        if not df_results.empty:
            success_df = df_results[df_results['success'] == True]
            if not success_df.empty and 'ari' in df_results.columns:
                best_ari = success_df.nlargest(5, 'ari')[['algorithm', 'dataset', 'ari', 'optimization']]
                logger.info("\nTop 5 algorithms by ARI score:")
                for _, row in best_ari.iterrows():
                    logger.info(f"  {row['algorithm']} on {row['dataset']} ({row['optimization']}): ARI = {row['ari']:.3f}")
        
        logger.info("=" * 80)
        logger.info(f"Detailed results saved to: {results_file}")
        logger.info(f"Summary report saved to: {summary_file}")
    
    def _create_summary_report(self, df_results: pd.DataFrame) -> Dict[str, Any]:
        """Create summary report from test results."""
        
        summary = {
            'test_info': {
                'timestamp': datetime.now().isoformat(),
                'total_tests': len(df_results),
                'successful_tests': int(df_results['success'].sum()),
                'failed_tests': int((~df_results['success']).sum()),
                'scale': 'memory'
            },
            'algorithm_performance': {},
            'dataset_difficulty': {},
            'optimization_impact': {}
        }
        
        # Algorithm performance analysis
        if not df_results.empty:
            for algorithm in df_results['algorithm'].unique():
                algo_results = df_results[df_results['algorithm'] == algorithm]
                summary['algorithm_performance'][algorithm] = {
                    'success_rate': float(algo_results['success'].mean()),
                    'avg_execution_time': float(algo_results['execution_time'].mean()),
                    'tested_datasets': list(algo_results['dataset'].unique())
                }
        
        # Dataset difficulty analysis
        for dataset in df_results['dataset'].unique():
            dataset_results = df_results[df_results['dataset'] == dataset]
            summary['dataset_difficulty'][dataset] = {
                'avg_success_rate': float(dataset_results['success'].mean()),
                'algorithms_tested': list(dataset_results['algorithm'].unique())
            }
        
        # Optimization impact
        if 'optimization' in df_results.columns:
            opt_comparison = df_results.groupby('optimization')['success'].mean()
            summary['optimization_impact'] = opt_comparison.to_dict()
        
        return summary

def main():
    """Main testing function."""
    
    # Setup
    tester = AlgorithmTester()
    
    print("Pattern Library Comprehensive Testing - Memory Scale")
    print("=" * 60)
    print("This test suite will:")
    print("1. Discover all implemented algorithms and metrics")
    print("2. Download benchmark datasets for all modalities")
    print("3. Generate synthetic datasets for comprehensive testing")
    print("4. Test algorithms with default and optimized hyperparameters")
    print("5. Generate detailed performance reports")
    print("=" * 60)
    
    try:
        # Run comprehensive tests
        tester.run_comprehensive_tests()
        
        print("\nTesting completed successfully!")
        print(f"Results saved in: {tester.results_dir}")
        
    except KeyboardInterrupt:
        logger.info("Testing interrupted by user")
        print("\nTesting interrupted. Partial results may be available.")
        
    except Exception as e:
        logger.error(f"Testing failed with error: {e}")
        logger.debug(traceback.format_exc())
        print(f"\nTesting failed: {e}")
        
    finally:
        # Save any partial results
        if tester.test_results:
            emergency_file = tester.results_dir / f"emergency_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(emergency_file, 'w') as f:
                json.dump(tester.test_results, f, indent=2)
            print(f"Emergency results saved to: {emergency_file}")

if __name__ == "__main__":
    main() 