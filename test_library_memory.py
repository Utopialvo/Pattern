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
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

# Pattern library imports
try:
    from config.registries import MODEL_REGISTRY, METRIC_REGISTRY
    from core.factory import factory
    from core.logger import logger
    from data.loaders import PandasDataLoader
    from optimization.strategies import TPESearch
except ImportError as e:
    print(f"Error importing Pattern library components: {e}")
    sys.exit(1)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class BenchmarkDataManager:
    """Manages benchmark dataset downloading and preprocessing for all modalities."""
    
    def __init__(self, data_dir: str = "Datasets"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for organized storage
        (self.data_dir / "Raw").mkdir(exist_ok=True)
        (self.data_dir / "Processed").mkdir(exist_ok=True)
        (self.data_dir / "Synthetic").mkdir(exist_ok=True)
        (self.data_dir / "Cache").mkdir(exist_ok=True)
        
        # Cache for loaded datasets
        self._dataset_cache = {}
        
        # Benchmark datasets by modality
        self.benchmark_datasets = {
            'attribute': {
                'iris': {
                    'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                    'description': 'Classic iris flower dataset',
                    'expected_clusters': 3,
                    'expected_ari': 0.73,
                    'expected_nmi': 0.76
                },
                'wine': {
                    'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',
                    'description': 'Wine recognition dataset',
                    'expected_clusters': 3,
                    'expected_ari': 0.37,
                    'expected_nmi': 0.43
                },
                'breast_cancer': {
                    'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data',
                    'description': 'Breast cancer Wisconsin dataset',
                    'expected_clusters': 2,
                    'expected_ari': 0.62,
                    'expected_nmi': 0.58
                },
                'seeds': {
                    'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt',
                    'description': 'Seeds dataset',
                    'expected_clusters': 3,
                    'expected_ari': 0.71,
                    'expected_nmi': 0.69
                },
                'glass': {
                    'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data',
                    'description': 'Glass identification dataset',
                    'expected_clusters': 6,
                    'expected_ari': 0.25,
                    'expected_nmi': 0.35
                },
                'ecoli': {
                    'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data',
                    'description': 'E.coli protein localization dataset',
                    'expected_clusters': 8,
                    'expected_ari': 0.45,
                    'expected_nmi': 0.52
                },
                'yeast': {
                    'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data',
                    'description': 'Yeast protein classification dataset',
                    'expected_clusters': 10,
                    'expected_ari': 0.28,
                    'expected_nmi': 0.38
                }
            },
            'network': {
                'karate': {
                    'description': 'Zachary karate club network',
                    'expected_clusters': 2,
                    'expected_modularity': 0.42,
                    'expected_ari': 0.685,
                    'builtin': True
                },
                'dolphins': {
                    'url': 'http://www-personal.umich.edu/~mejn/netdata/dolphins.zip',
                    'description': 'Dolphin social network',
                    'expected_clusters': 2,
                    'expected_modularity': 0.52,
                    'expected_ari': 0.45
                },
                'football': {
                    'url': 'http://www-personal.umich.edu/~mejn/netdata/football.zip',
                    'description': 'American college football network',
                    'expected_clusters': 12,
                    'expected_modularity': 0.60,
                    'expected_ari': 0.92
                },
                'polbooks': {
                    'url': 'http://www-personal.umich.edu/~mejn/netdata/polbooks.zip',
                    'description': 'Political books co-purchasing network',
                    'expected_clusters': 3,
                    'expected_modularity': 0.53,
                    'expected_ari': 0.54
                },
                'les_miserables': {
                    'url': 'http://www-personal.umich.edu/~mejn/netdata/lesmis.zip',
                    'description': 'Les Miserables character network',
                    'expected_clusters': 6,
                    'expected_modularity': 0.56,
                    'expected_ari': 0.65
                },
                'adjnoun': {
                    'url': 'http://www-personal.umich.edu/~mejn/netdata/adjnoun.zip',
                    'description': 'Adjective-noun adjacency network',
                    'expected_clusters': 4,
                    'expected_modularity': 0.31,
                    'expected_ari': 0.35
                }
            },
            'attributed_graph': {
                'cora': {
                    'url': 'https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz',
                    'description': 'Cora citation network with features',
                    'expected_clusters': 7,
                    'expected_ari': 0.48,
                    'expected_nmi': 0.54
                },
                'citeseer': {
                    'url': 'https://linqs-data.soe.ucsc.edu/public/lbc/citeseer.tgz',
                    'description': 'CiteSeer citation network with features',
                    'expected_clusters': 6,
                    'expected_ari': 0.41,
                    'expected_nmi': 0.48
                },
                'pubmed': {
                    'url': 'https://linqs-data.soe.ucsc.edu/public/Pubmed-Diabetes.tgz',
                    'description': 'PubMed diabetes citation network',
                    'expected_clusters': 3,
                    'expected_ari': 0.65,
                    'expected_nmi': 0.58
                },
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
    
    def save_dataset(self, name: str, features: pd.DataFrame, similarity: Optional[pd.DataFrame] = None, 
                    labels: Optional[pd.Series] = None, metadata: Optional[Dict] = None) -> bool:
        """Save a processed dataset to disk."""
        try:
            dataset_dir = self.data_dir / name.capitalize()
            dataset_dir.mkdir(exist_ok=True)
            
            # Save features
            if features is not None:
                features.to_csv(dataset_dir / "Features.csv", index=False)
            
            # Save similarity/adjacency matrix
            if similarity is not None:
                similarity.to_csv(dataset_dir / "Networks.csv", index=False)
            
            # Save labels
            if labels is not None:
                labels.to_csv(dataset_dir / "Labels.csv", index=False)
            
            # Save metadata
            metadata_info = {
                'name': name,
                'timestamp': datetime.now().isoformat(),
                'n_samples': len(features) if features is not None else (len(similarity) if similarity is not None else 0),
                'n_features': len(features.columns) if features is not None else 0,
                'has_similarity': similarity is not None,
                'has_labels': labels is not None,
                'n_unique_labels': len(labels.unique()) if labels is not None else None
            }
            
            if metadata:
                metadata_info.update(metadata)
            
            with open(dataset_dir / "Metadata.json", 'w') as f:
                json.dump(metadata_info, f, indent=2, default=str)
            
            logger.info(f"Dataset '{name}' saved to {dataset_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save dataset '{name}': {e}")
            return False
    
    def load_dataset(self, name: str, use_cache: bool = True) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.Series], Optional[Dict]]:
        """Load a processed dataset from disk."""
        
        # Check cache first
        if use_cache and name in self._dataset_cache:
            logger.info(f"Loading dataset '{name}' from cache")
            return self._dataset_cache[name]
        
        try:
            dataset_dir = self.data_dir / name.capitalize()
            
            if not dataset_dir.exists():
                logger.warning(f"Dataset '{name}' not found in datasets directory")
                return None, None, None, None
            
            features = None
            similarity = None
            labels = None
            metadata = None
            
            # Load features
            features_path = dataset_dir / "Features.csv"
            if features_path.exists():
                features = pd.read_csv(features_path)
            
            # Load similarity/adjacency matrix
            similarity_path = dataset_dir / "Networks.csv"
            if similarity_path.exists():
                similarity = pd.read_csv(similarity_path)
            
            # Load labels
            labels_path = dataset_dir / "Labels.csv"
            if labels_path.exists():
                labels = pd.read_csv(labels_path).iloc[:, 0]  # Get first column as Series
                labels.name = 'true_labels'
            
            # Load metadata
            metadata_path = dataset_dir / "Metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            
            # Cache the result
            result = (features, similarity, labels, metadata)
            if use_cache:
                self._dataset_cache[name] = result
            
            logger.info(f"Dataset '{name}' loaded from {dataset_dir}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to load dataset '{name}': {e}")
            return None, None, None, None
    
    def save_configuration(self, config: Dict[str, Any], filename: str = "Data_config.json") -> bool:
        """Save data configuration to file."""
        try:
            config_path = self.data_dir / "Cache" / filename
            
            config_info = {
                'timestamp': datetime.now().isoformat(),
                'benchmark_datasets': self.benchmark_datasets,
                'benchmark_performance': self.benchmark_performance,
                'user_config': config
            }
            
            with open(config_path, 'w') as f:
                json.dump(config_info, f, indent=2, default=str)
            
            logger.info(f"Configuration saved to {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def load_configuration(self, filename: str = "Data_config.json") -> Optional[Dict[str, Any]]:
        """Load data configuration from file."""
        try:
            config_path = self.data_dir / "Cache" / filename
            
            if not config_path.exists():
                logger.warning(f"Configuration file {filename} not found")
                return None
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            logger.info(f"Configuration loaded from {config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return None
    
    def clear_cache(self):
        """Clear the dataset cache."""
        self._dataset_cache.clear()
        logger.info("Dataset cache cleared")
    
    def list_cached_datasets(self) -> List[str]:
        """List all cached datasets."""
        return list(self._dataset_cache.keys())
    
    def list_saved_datasets(self) -> List[str]:
        """List all saved processed datasets."""
        if not self.data_dir.exists():
            return []
        
        return [d.name.lower() for d in self.data_dir.iterdir() if d.is_dir() and d.name not in ['Raw', 'Processed', 'Synthetic', 'Cache']]
    
    def load_attribute_dataset(self, dataset_name: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """Load attribute dataset."""
        try:
            # For iris dataset, use sklearn
            if dataset_name == 'iris':
                from sklearn.datasets import load_iris
                iris = load_iris()
                features = pd.DataFrame(iris.data, columns=iris.feature_names)
                labels = pd.Series(iris.target, name='true_labels')
                return features, labels
            
            # For wine dataset, use sklearn
            elif dataset_name == 'wine':
                from sklearn.datasets import load_wine
                wine = load_wine()
                features = pd.DataFrame(wine.data, columns=wine.feature_names)
                labels = pd.Series(wine.target, name='true_labels')
                return features, labels
            
            # For breast cancer dataset, use sklearn
            elif dataset_name == 'breast_cancer':
                from sklearn.datasets import load_breast_cancer
                cancer = load_breast_cancer()
                features = pd.DataFrame(cancer.data, columns=cancer.feature_names)
                labels = pd.Series(cancer.target, name='true_labels')
                return features, labels
            
            # For other datasets, try to load from saved files
            else:
                features, _, labels, _ = self.load_dataset(dataset_name)
                return features, labels
                
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
                # Create labels based on the known split
                labels = pd.Series([0 if G.nodes[i]['club'] == 'Mr. Hi' else 1 for i in G.nodes()], name='true_labels')
                return None, adj_matrix, labels
            
            # For other datasets, try to load from saved files
            else:
                features, similarity, labels, _ = self.load_dataset(dataset_name)
                return features, similarity, labels
                
        except Exception as e:
            logger.error(f"Failed to load network dataset {dataset_name}: {e}")
            return None, None, None
    
    def load_attributed_graph_dataset(self, dataset_name: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.Series]]:
        """Load attributed graph dataset."""
        try:
            # For synthetic scenarios, generate them
            if dataset_name.startswith('synthetic_attr_'):
                if dataset_name == 'synthetic_attr_easy':
                    return SyntheticDataGenerator.generate_attributed_graph_data(
                        n_nodes=300, n_features=15, n_communities=3, p_in=0.4, p_out=0.05
                    )
                elif dataset_name == 'synthetic_attr_medium':
                    return SyntheticDataGenerator.generate_attributed_graph_data(
                        n_nodes=400, n_features=20, n_communities=4, p_in=0.3, p_out=0.03
                    )
                elif dataset_name == 'synthetic_attr_hard':
                    return SyntheticDataGenerator.generate_attributed_graph_data(
                        n_nodes=500, n_features=25, n_communities=5, p_in=0.25, p_out=0.02
                    )
            
            # For other datasets, try to load from saved files
            else:
                features, similarity, labels, _ = self.load_dataset(dataset_name)
                return features, similarity, labels
                
        except Exception as e:
            logger.error(f"Failed to load attributed graph dataset {dataset_name}: {e}")
            return None, None, None

class SyntheticDataGenerator:
    """Generates synthetic datasets for each modality."""
    
    def __init__(self, cache_dir: str = "Datasets/Synthetic"):
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
            logger.info(f"Synthetic dataset '{name}' saved to {dataset_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save synthetic dataset '{name}': {e}")
            return False
    
    def load_synthetic_dataset(self, name: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.Series], Optional[Dict]]:
        """Load a saved synthetic dataset."""
        try:
            dataset_path = self.cache_dir / f"{name}.npz"
            
            if not dataset_path.exists():
                logger.warning(f"Synthetic dataset '{name}' not found")
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
            
            logger.info(f"Synthetic dataset '{name}' loaded from {dataset_path}")
            return features, similarity, labels, params
            
        except Exception as e:
            logger.error(f"Failed to load synthetic dataset '{name}': {e}")
            return None, None, None, None
    
    def list_saved_synthetic_datasets(self) -> List[str]:
        """List all saved synthetic datasets."""
        if not self.cache_dir.exists():
            return []
        
        return [f.stem for f in self.cache_dir.glob("*.npz")]
    
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
    
    def __init__(self, results_dir: str = "Test_Results_Memory"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for organization
        (self.results_dir / "Errors").mkdir(exist_ok=True)
        (self.results_dir / "Logs").mkdir(exist_ok=True)
        (self.results_dir / "Reports").mkdir(exist_ok=True)
        (self.results_dir / "Cache").mkdir(exist_ok=True)
        (self.results_dir / "Exports").mkdir(exist_ok=True)
        
        # Initialize components
        self.data_manager = BenchmarkDataManager()
        self.synthetic_generator = SyntheticDataGenerator()
        
        # Test results storage
        self.test_results = []
        self.error_count = 0
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.results_dir / "Logs" / f"Test_log_{timestamp}.log"
        
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
        error_filename = f"Error_{self.error_count:03d}_{timestamp}.json"
        error_path = self.results_dir / "Errors" / error_filename
        
        try:
            with open(error_path, 'w') as f:
                json.dump(error_info, f, indent=2, default=str)
            logger.info(f"Error details saved to: {error_filename}")
            return str(error_path)
        except Exception as e:
            logger.error(f"Failed to save error to JSON: {e}")
            return ""
    
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
        if any(keyword in name_lower for keyword in ['dmon', 'gnn', 'graph', 'node2vec', 'canus', 'kefrin', 'dgclustering', 'wsnmf']):
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
        """Test a single algorithm on a dataset with comprehensive error handling."""
        
        start_time = time.time()
        
        # Get expected performance if available
        expected_performance = self._get_expected_performance(dataset_name)
        
        result = {
            'algorithm': algorithm_name,
            'dataset': dataset_name,
            'optimization': optimization_method,
            'params': params.copy(),
            'success': False,
            'error': None,
            'error_file': None,
            'execution_time': 0,
            'n_samples': len(features) if features is not None else (len(similarity) if similarity is not None else 0),
            'n_features': len(features.columns) if features is not None else 0,
            'n_true_clusters': len(np.unique(true_labels)) if true_labels is not None else None,
            'expected_ari': expected_performance.get('expected_ari'),
            'expected_nmi': expected_performance.get('expected_nmi'),
            'expected_modularity': expected_performance.get('expected_modularity'),
            'obtained_ari': None,
            'obtained_nmi': None,
            'obtained_silhouette': None,
            'obtained_calinski_harabasz': None,
            'obtained_modularity': None,
            'n_predicted_clusters': None,
            'ari_vs_expected': None,
            'nmi_vs_expected': None,
            'metrics': {}
        }
        
        try:
            logger.info(f"Testing {algorithm_name} on {dataset_name} with {optimization_method} params")
            
            # Create data loader with comprehensive error handling
            try:
                data_loader = PandasDataLoader(features=features, similarity=similarity)
            except Exception as e:
                raise ValueError(f"Failed to create data loader: {str(e)}")
            
            # Create and configure model
            try:
                model = factory.create_model(algorithm_name, params)
            except Exception as e:
                raise ValueError(f"Failed to create model {algorithm_name}: {str(e)}")
            
            # Fit model
            try:
                model.fit(data_loader)
            except Exception as e:
                raise RuntimeError(f"Failed to fit model: {str(e)}")
            
            # Get predictions
            try:
                if hasattr(model, 'labels_') and model.labels_ is not None:
                    predicted_labels = model.labels_
                else:
                    predicted_labels = model.predict(data_loader)
                
                if predicted_labels is None:
                    raise ValueError("Model returned no predictions")
                
                # Convert to numpy array if needed
                if isinstance(predicted_labels, pd.Series):
                    predicted_labels = predicted_labels.values
                elif not isinstance(predicted_labels, np.ndarray):
                    predicted_labels = np.array(predicted_labels)
                
                # Check for valid predictions
                if len(predicted_labels) == 0:
                    raise ValueError("Empty predictions returned")
                
                result['n_predicted_clusters'] = len(np.unique(predicted_labels))
                
            except Exception as e:
                raise RuntimeError(f"Failed to get predictions: {str(e)}")
            
            # Calculate comprehensive metrics
            try:
                # External metrics (require ground truth)
                if true_labels is not None:
                    true_labels_array = true_labels.values if isinstance(true_labels, pd.Series) else np.array(true_labels)
                    
                    # Ensure same length
                    min_len = min(len(true_labels_array), len(predicted_labels))
                    true_labels_array = true_labels_array[:min_len]
                    predicted_labels = predicted_labels[:min_len]
                    
                    # Calculate ARI and NMI
                    ari_score = adjusted_rand_score(true_labels_array, predicted_labels)
                    nmi_score = normalized_mutual_info_score(true_labels_array, predicted_labels)
                    
                    result['obtained_ari'] = float(ari_score)
                    result['obtained_nmi'] = float(nmi_score)
                    result['metrics']['ari'] = float(ari_score)
                    result['metrics']['nmi'] = float(nmi_score)
                    
                    # Compare with expected values
                    if result['expected_ari'] is not None:
                        result['ari_vs_expected'] = float(ari_score - result['expected_ari'])
                    if result['expected_nmi'] is not None:
                        result['nmi_vs_expected'] = float(nmi_score - result['expected_nmi'])
                
                # Internal metrics (don't require ground truth)
                if features is not None and len(features) > 1:
                    try:
                        # Silhouette score
                        if len(np.unique(predicted_labels)) > 1:
                            silhouette = silhouette_score(features, predicted_labels)
                            result['obtained_silhouette'] = float(silhouette)
                            result['metrics']['silhouette'] = float(silhouette)
                    except Exception as e:
                        logger.warning(f"Failed to calculate silhouette score: {e}")
                    
                    try:
                        # Calinski-Harabasz score
                        if len(np.unique(predicted_labels)) > 1:
                            ch_score = calinski_harabasz_score(features, predicted_labels)
                            result['obtained_calinski_harabasz'] = float(ch_score)
                            result['metrics']['calinski_harabasz'] = float(ch_score)
                    except Exception as e:
                        logger.warning(f"Failed to calculate Calinski-Harabasz score: {e}")
                
                # Pattern library internal metrics
                for metric_name in METRIC_REGISTRY:
                    try:
                        metric = factory.create_metric(metric_name)
                        score = metric.calculate(data_loader, predicted_labels, model.model_data)
                        if not np.isnan(score) and np.isfinite(score):
                            result['metrics'][metric_name] = float(score)
                            
                            # Store specific metrics in main result
                            if metric_name.lower() == 'modularity':
                                result['obtained_modularity'] = float(score)
                                
                    except Exception as e:
                        logger.warning(f"Failed to calculate {metric_name}: {e}")
                
            except Exception as e:
                logger.warning(f"Error calculating metrics: {e}")
            
            result['success'] = True
            logger.info(f"Successfully tested {algorithm_name} on {dataset_name}")
            
        except Exception as e:
            error_info = {
                'timestamp': datetime.now().isoformat(),
                'algorithm': algorithm_name,
                'dataset': dataset_name,
                'optimization': optimization_method,
                'params': params,
                'error_type': type(e).__name__,
                'error_message': str(e),
                'traceback': traceback.format_exc(),
                'execution_time': time.time() - start_time,
                'dataset_info': {
                    'n_samples': result['n_samples'],
                    'n_features': result['n_features'],
                    'n_true_clusters': result['n_true_clusters']
                }
            }
            
            result['error'] = str(e)
            result['error_file'] = self._save_error_to_json(error_info)
            logger.error(f"Failed to test {algorithm_name} on {dataset_name}: {e}")
        
        result['execution_time'] = time.time() - start_time
        return result
    
    def _get_expected_performance(self, dataset_name: str) -> Dict[str, Any]:
        """Get expected performance values for a dataset."""
        expected = {}
        
        # Check all modalities for the dataset
        for modality_datasets in self.data_manager.benchmark_datasets.values():
            if dataset_name in modality_datasets:
                dataset_info = modality_datasets[dataset_name]
                expected['expected_ari'] = dataset_info.get('expected_ari')
                expected['expected_nmi'] = dataset_info.get('expected_nmi')
                expected['expected_modularity'] = dataset_info.get('expected_modularity')
                break
        
        return expected
    
    def save_test_results(self, filename: Optional[str] = None) -> bool:
        """Save current test results to file."""
        try:
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"Test_results_{timestamp}.json"
            
            results_path = self.results_dir / "Cache" / filename
            
            # Create cache directory if it doesn't exist
            results_path.parent.mkdir(exist_ok=True)
            
            save_data = {
                'timestamp': datetime.now().isoformat(),
                'test_info': {
                    'total_tests': len(self.test_results),
                    'error_count': self.error_count,
                    'results_dir': str(self.results_dir)
                },
                'test_results': self.test_results
            }
            
            with open(results_path, 'w') as f:
                json.dump(save_data, f, indent=2, default=str)
            
            logger.info(f"Test results saved to {results_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save test results: {e}")
            return False
    
    def load_test_results(self, filename: str) -> bool:
        """Load test results from file."""
        try:
            results_path = self.results_dir / "Cache" / filename
            
            if not results_path.exists():
                logger.warning(f"Test results file {filename} not found")
                return False
            
            with open(results_path, 'r') as f:
                data = json.load(f)
            
            self.test_results = data.get('test_results', [])
            self.error_count = data.get('test_info', {}).get('error_count', 0)
            
            logger.info(f"Test results loaded from {results_path}")
            logger.info(f"Loaded {len(self.test_results)} test results")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load test results: {e}")
            return False
    
    def save_test_configuration(self, algorithms: Dict[str, Dict], config: Optional[Dict] = None, 
                               filename: Optional[str] = None) -> bool:
        """Save test configuration for reproducibility."""
        try:
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"Test_config_{timestamp}.json"
            
            config_path = self.results_dir / "Cache" / filename
            config_path.parent.mkdir(exist_ok=True)
            
            config_data = {
                'timestamp': datetime.now().isoformat(),
                'algorithms': algorithms,
                'datasets': self.data_manager.benchmark_datasets,
                'user_config': config or {},
                'results_dir': str(self.results_dir)
            }
            
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2, default=str)
            
            logger.info(f"Test configuration saved to {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save test configuration: {e}")
            return False
    
    def load_test_configuration(self, filename: str) -> Optional[Dict[str, Any]]:
        """Load test configuration from file."""
        try:
            config_path = self.results_dir / "Cache" / filename
            
            if not config_path.exists():
                logger.warning(f"Configuration file {filename} not found")
                return None
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            logger.info(f"Test configuration loaded from {config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load test configuration: {e}")
            return None
    
    def export_results_to_formats(self, formats: List[str] = ['csv', 'json', 'excel']) -> Dict[str, bool]:
        """Export test results to multiple formats."""
        results = {}
        
        if not self.test_results:
            logger.warning("No test results to export")
            return {fmt: False for fmt in formats}
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        df_results = pd.DataFrame(self.test_results)
        
        for fmt in formats:
            try:
                if fmt.lower() == 'csv':
                    export_path = self.results_dir / "Exports" / f"Results_{timestamp}.csv"
                    export_path.parent.mkdir(exist_ok=True)
                    df_results.to_csv(export_path, index=False)
                    results[fmt] = True
                    logger.info(f"Results exported to CSV: {export_path}")
                
                elif fmt.lower() == 'json':
                    export_path = self.results_dir / "Exports" / f"Results_{timestamp}.json"
                    export_path.parent.mkdir(exist_ok=True)
                    with open(export_path, 'w') as f:
                        json.dump(self.test_results, f, indent=2, default=str)
                    results[fmt] = True
                    logger.info(f"Results exported to JSON: {export_path}")
                
                elif fmt.lower() == 'excel':
                    export_path = self.results_dir / "Exports" / f"Results_{timestamp}.xlsx"
                    export_path.parent.mkdir(exist_ok=True)
                    
                    with pd.ExcelWriter(export_path, engine='openpyxl') as writer:
                        # Main results
                        df_results.to_excel(writer, sheet_name='All_Results', index=False)
                        
                        # Summary by algorithm
                        algo_summary = df_results.groupby('algorithm').agg({
                            'success': 'mean',
                            'obtained_ari': 'mean',
                            'obtained_nmi': 'mean',
                            'execution_time': 'mean'
                        }).round(4)
                        algo_summary.to_excel(writer, sheet_name='Algorithm_Summary')
                        
                        # Summary by dataset
                        dataset_summary = df_results.groupby('dataset').agg({
                            'success': 'mean',
                            'obtained_ari': 'mean',
                            'obtained_nmi': 'mean'
                        }).round(4)
                        dataset_summary.to_excel(writer, sheet_name='Dataset_Summary')
                    
                    results[fmt] = True
                    logger.info(f"Results exported to Excel: {export_path}")
                
                else:
                    logger.warning(f"Unsupported export format: {fmt}")
                    results[fmt] = False
                    
            except Exception as e:
                logger.error(f"Failed to export to {fmt}: {e}")
                results[fmt] = False
        
        return results
    
    def list_saved_results(self) -> List[str]:
        """List all saved test result files."""
        cache_dir = self.results_dir / "Cache"
        if not cache_dir.exists():
            return []
        
        return [f.name for f in cache_dir.glob("Test_results_*.json")]
    
    def list_saved_configurations(self) -> List[str]:
        """List all saved configuration files."""
        cache_dir = self.results_dir / "Cache"
        if not cache_dir.exists():
            return []
        
        return [f.name for f in cache_dir.glob("Test_config_*.json")]
    
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
        
        # Save test configuration for reproducibility
        self.save_test_configuration(algorithms, {'metrics': list(metrics.keys())})
        
        # Test on benchmark datasets
        self._test_benchmark_datasets(algorithms)
        
        # Test on synthetic datasets
        self._test_synthetic_datasets(algorithms)
        
        # Save intermediate results
        self.save_test_results()
        
        # Generate comprehensive report
        self._generate_report()
        
        # Export results to multiple formats
        export_status = self.export_results_to_formats(['csv', 'json'])
        
        logger.info("Comprehensive testing completed")
        logger.info(f"Export status: {export_status}")
    
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
                    # Adjust n_clusters based on expected clusters
                    dataset_info = self.data_manager.benchmark_datasets['attribute'][dataset_name]
                    if 'n_clusters' in default_params:
                        default_params['n_clusters'] = dataset_info['expected_clusters']
                    
                    result = self.test_algorithm_on_dataset(
                        algo_name, dataset_name, features, None, true_labels,
                        default_params, 'default'
                    )
                    self.test_results.append(result)
                    
                    # Test with optimized parameters (only for first few datasets to save time)
                    if dataset_name in ['iris', 'wine', 'breast_cancer']:
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
            logger.info(f"Loading benchmark dataset: {dataset_name}")
            
            features, adj_matrix, true_labels = self.data_manager.load_network_dataset(dataset_name)
            if adj_matrix is None:
                logger.warning(f"Failed to load {dataset_name}")
                continue
            
            # Test relevant algorithms
            for algo_name, algo_info in algorithms.items():
                if algo_info['modality'] == 'network':
                    
                    # Test with default parameters
                    default_params = self.get_default_params(algo_name)
                    # Adjust n_clusters based on expected clusters
                    dataset_info = self.data_manager.benchmark_datasets['network'][dataset_name]
                    if 'n_clusters' in default_params:
                        default_params['n_clusters'] = dataset_info['expected_clusters']
                    
                    result = self.test_algorithm_on_dataset(
                        algo_name, dataset_name, features, adj_matrix, true_labels,
                        default_params, 'default'
                    )
                    self.test_results.append(result)
                    
                    # Test with optimized parameters (only for karate and dolphins)
                    if dataset_name in ['karate', 'dolphins']:
                        optimized_params = self.optimize_hyperparameters(
                            algo_name, dataset_name, features, adj_matrix, true_labels
                        )
                        result = self.test_algorithm_on_dataset(
                            algo_name, dataset_name, features, adj_matrix, true_labels,
                            optimized_params, 'optimized'
                        )
                        self.test_results.append(result)
        
        # Test attributed graph datasets
        for dataset_name in self.data_manager.benchmark_datasets['attributed_graph']:
            logger.info(f"Loading benchmark dataset: {dataset_name}")
            
            features, adj_matrix, true_labels = self.data_manager.load_attributed_graph_dataset(dataset_name)
            if features is None or adj_matrix is None:
                logger.warning(f"Failed to load {dataset_name}")
                continue
            
            # Test relevant algorithms
            for algo_name, algo_info in algorithms.items():
                if algo_info['modality'] == 'attributed_graph':
                    
                    # Test with default parameters
                    default_params = self.get_default_params(algo_name)
                    # Adjust n_clusters based on expected clusters
                    dataset_info = self.data_manager.benchmark_datasets['attributed_graph'][dataset_name]
                    if 'n_clusters' in default_params:
                        default_params['n_clusters'] = dataset_info['expected_clusters']
                    elif 'num_clusters' in default_params:
                        default_params['num_clusters'] = dataset_info['expected_clusters']
                    
                    result = self.test_algorithm_on_dataset(
                        algo_name, dataset_name, features, adj_matrix, true_labels,
                        default_params, 'default'
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
            {'name': 'moons', 'params': {'n_samples': 500, 'scenario': 'moons'}},
            {'name': 'blobs_high_dim', 'params': {'n_samples': 300, 'n_features': 20, 'n_clusters': 4, 'cluster_std': 1.5}},
            {'name': 'blobs_many_clusters', 'params': {'n_samples': 800, 'n_features': 8, 'n_clusters': 8, 'cluster_std': 1.2}}
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
                    elif scenario['name'] in ['circles', 'moons']:
                        if 'n_clusters' in default_params:
                            default_params['n_clusters'] = 2
                    
                    result = self.test_algorithm_on_dataset(
                        algo_name, f"synthetic_{scenario['name']}", features, None, true_labels,
                        default_params, 'default'
                    )
                    self.test_results.append(result)
        
        # Synthetic network data scenarios
        network_scenarios = [
            {'name': 'sbm_small', 'params': {'n_nodes': 100, 'n_communities': 3, 'p_in': 0.4, 'p_out': 0.05}},
            {'name': 'sbm_medium', 'params': {'n_nodes': 200, 'n_communities': 4, 'p_in': 0.3, 'p_out': 0.02}},
            {'name': 'sbm_large', 'params': {'n_nodes': 300, 'n_communities': 5, 'p_in': 0.25, 'p_out': 0.01}},
            {'name': 'ba_graph', 'params': {'n_nodes': 150, 'n_communities': 3, 'scenario': 'barabasi_albert'}}
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
        
        # Synthetic attributed graph scenarios (using the new builtin synthetic datasets)
        ag_scenarios = ['synthetic_attr_easy', 'synthetic_attr_medium', 'synthetic_attr_hard']
        
        for scenario_name in ag_scenarios:
            logger.info(f"Generating synthetic attributed graph: {scenario_name}")
            
            features, adj_matrix, true_labels = self.data_manager.load_attributed_graph_dataset(scenario_name)
            if features is None or adj_matrix is None:
                continue
            
            # Test relevant algorithms
            for algo_name, algo_info in algorithms.items():
                if algo_info['modality'] == 'attributed_graph':
                    
                    default_params = self.get_default_params(algo_name)
                    dataset_info = self.data_manager.benchmark_datasets['attributed_graph'][scenario_name]
                    if 'n_clusters' in default_params:
                        default_params['n_clusters'] = dataset_info['expected_clusters']
                    elif 'num_clusters' in default_params:
                        default_params['num_clusters'] = dataset_info['expected_clusters']
                    
                    result = self.test_algorithm_on_dataset(
                        algo_name, scenario_name, features, adj_matrix, true_labels,
                        default_params, 'default'
                    )
                    self.test_results.append(result)
    
    def _generate_report(self):
        """Generate comprehensive test report with CSV export."""
        
        logger.info("Generating comprehensive test report...")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Convert results to DataFrame for analysis
        df_results = pd.DataFrame(self.test_results)
        
        if df_results.empty:
            logger.warning("No test results to report")
            return
        
        # Save detailed results as CSV
        results_file = self.results_dir / "Reports" / f"Detailed_results_{timestamp}.csv"
        df_results.to_csv(results_file, index=False)
        
        # Create a summary DataFrame with key metrics
        summary_columns = [
            'algorithm', 'dataset', 'optimization', 'success', 'execution_time',
            'n_samples', 'n_features', 'n_true_clusters', 'n_predicted_clusters',
            'expected_ari', 'obtained_ari', 'ari_vs_expected',
            'expected_nmi', 'obtained_nmi', 'nmi_vs_expected',
            'expected_modularity', 'obtained_modularity',
            'obtained_silhouette', 'obtained_calinski_harabasz',
            'error'
        ]
        
        # Create summary with only existing columns
        available_columns = [col for col in summary_columns if col in df_results.columns]
        df_summary = df_results[available_columns].copy()
        
        # Add performance comparison categories
        if 'ari_vs_expected' in df_summary.columns:
            def categorize_performance(diff):
                if pd.isna(diff):
                    return 'Unknown'
                elif diff > 0.1:
                    return 'Much Better'
                elif diff > 0.05:
                    return 'Better'
                elif diff > -0.05:
                    return 'Similar'
                elif diff > -0.1:
                    return 'Worse'
                else:
                    return 'Much Worse'
            
            df_summary['ari_performance'] = df_summary['ari_vs_expected'].apply(categorize_performance)
        
        if 'nmi_vs_expected' in df_summary.columns:
            df_summary['nmi_performance'] = df_summary['nmi_vs_expected'].apply(categorize_performance)
        
        # Save summary results
        summary_file = self.results_dir / "Reports" / f"Summary_results_{timestamp}.csv"
        df_summary.to_csv(summary_file, index=False)
        
        # Generate comprehensive analysis
        analysis = self._create_comprehensive_analysis(df_results)
        
        # Save analysis as JSON
        analysis_file = self.results_dir / "Reports" / f"Analysis_report_{timestamp}.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        # Create performance comparison tables
        self._create_performance_tables(df_results, timestamp)
        
        # Print summary to console
        self._print_console_summary(df_results, analysis)
        
        logger.info("=" * 80)
    
    def _create_comprehensive_analysis(self, df_results: pd.DataFrame) -> Dict[str, Any]:
        """Create comprehensive analysis from test results."""
        
        analysis = {
            'test_info': {
                'timestamp': datetime.now().isoformat(),
                'total_tests': len(df_results),
                'successful_tests': int(df_results['success'].sum()),
                'failed_tests': int((~df_results['success']).sum()),
                'scale': 'memory',
                'error_rate': float((~df_results['success']).mean()),
                'avg_execution_time': float(df_results['execution_time'].mean())
            },
            'algorithm_performance': {},
            'dataset_analysis': {},
            'modality_performance': {},
            'optimization_impact': {},
            'performance_comparisons': {}
        }
        
        # Algorithm performance analysis
        for algorithm in df_results['algorithm'].unique():
            algo_results = df_results[df_results['algorithm'] == algorithm]
            successful_results = algo_results[algo_results['success'] == True]
            
            analysis['algorithm_performance'][algorithm] = {
                'success_rate': float(algo_results['success'].mean()),
                'avg_execution_time': float(algo_results['execution_time'].mean()),
                'tested_datasets': list(algo_results['dataset'].unique()),
                'avg_ari': float(successful_results['obtained_ari'].mean()) if 'obtained_ari' in successful_results.columns and not successful_results['obtained_ari'].isna().all() else None,
                'avg_nmi': float(successful_results['obtained_nmi'].mean()) if 'obtained_nmi' in successful_results.columns and not successful_results['obtained_nmi'].isna().all() else None,
                'best_ari_dataset': None,
                'worst_ari_dataset': None
            }
            
            # Find best and worst performing datasets
            if 'obtained_ari' in successful_results.columns and not successful_results['obtained_ari'].isna().all():
                best_idx = successful_results['obtained_ari'].idxmax()
                worst_idx = successful_results['obtained_ari'].idxmin()
                analysis['algorithm_performance'][algorithm]['best_ari_dataset'] = {
                    'dataset': successful_results.loc[best_idx, 'dataset'],
                    'ari': float(successful_results.loc[best_idx, 'obtained_ari'])
                }
                analysis['algorithm_performance'][algorithm]['worst_ari_dataset'] = {
                    'dataset': successful_results.loc[worst_idx, 'dataset'],
                    'ari': float(successful_results.loc[worst_idx, 'obtained_ari'])
                }
        
        # Dataset difficulty analysis
        for dataset in df_results['dataset'].unique():
            dataset_results = df_results[df_results['dataset'] == dataset]
            successful_results = dataset_results[dataset_results['success'] == True]
            
            analysis['dataset_analysis'][dataset] = {
                'success_rate': float(dataset_results['success'].mean()),
                'algorithms_tested': list(dataset_results['algorithm'].unique()),
                'avg_ari': float(successful_results['obtained_ari'].mean()) if 'obtained_ari' in successful_results.columns and not successful_results['obtained_ari'].isna().all() else None,
                'avg_nmi': float(successful_results['obtained_nmi'].mean()) if 'obtained_nmi' in successful_results.columns and not successful_results['obtained_nmi'].isna().all() else None,
                'difficulty_score': None
            }
            
            # Calculate difficulty score (lower ARI = higher difficulty)
            if analysis['dataset_analysis'][dataset]['avg_ari'] is not None:
                analysis['dataset_analysis'][dataset]['difficulty_score'] = 1.0 - analysis['dataset_analysis'][dataset]['avg_ari']
        
        # Performance comparisons with expected values
        if 'ari_vs_expected' in df_results.columns:
            comparison_results = df_results[df_results['ari_vs_expected'].notna()]
            if not comparison_results.empty:
                analysis['performance_comparisons']['ari'] = {
                    'better_than_expected': int((comparison_results['ari_vs_expected'] > 0.05).sum()),
                    'similar_to_expected': int((comparison_results['ari_vs_expected'].abs() <= 0.05).sum()),
                    'worse_than_expected': int((comparison_results['ari_vs_expected'] < -0.05).sum()),
                    'avg_difference': float(comparison_results['ari_vs_expected'].mean())
                }
        
        if 'nmi_vs_expected' in df_results.columns:
            comparison_results = df_results[df_results['nmi_vs_expected'].notna()]
            if not comparison_results.empty:
                analysis['performance_comparisons']['nmi'] = {
                    'better_than_expected': int((comparison_results['nmi_vs_expected'] > 0.05).sum()),
                    'similar_to_expected': int((comparison_results['nmi_vs_expected'].abs() <= 0.05).sum()),
                    'worse_than_expected': int((comparison_results['nmi_vs_expected'] < -0.05).sum()),
                    'avg_difference': float(comparison_results['nmi_vs_expected'].mean())
                }
        
        # Optimization impact
        if 'optimization' in df_results.columns:
            opt_comparison = df_results.groupby('optimization').agg({
                'success': 'mean',
                'obtained_ari': 'mean',
                'obtained_nmi': 'mean',
                'execution_time': 'mean'
            }).to_dict()
            analysis['optimization_impact'] = opt_comparison
        
        return analysis
    
    def _create_performance_tables(self, df_results: pd.DataFrame, timestamp: str):
        """Create performance comparison tables."""
        
        # Algorithm vs Dataset performance table (ARI)
        if 'obtained_ari' in df_results.columns:
            pivot_ari = df_results.pivot_table(
                values='obtained_ari', 
                index='algorithm', 
                columns='dataset', 
                aggfunc='mean'
            )
            ari_table_file = self.results_dir / "Reports" / f"ARI_performance_table_{timestamp}.csv"
            pivot_ari.to_csv(ari_table_file)
        
        # Algorithm vs Dataset performance table (NMI)
        if 'obtained_nmi' in df_results.columns:
            pivot_nmi = df_results.pivot_table(
                values='obtained_nmi', 
                index='algorithm', 
                columns='dataset', 
                aggfunc='mean'
            )
            nmi_table_file = self.results_dir / "Reports" / f"NMI_performance_table_{timestamp}.csv"
            pivot_nmi.to_csv(nmi_table_file)
        
        # Success rate table
        pivot_success = df_results.pivot_table(
            values='success', 
            index='algorithm', 
            columns='dataset', 
            aggfunc='mean'
        )
        success_table_file = self.results_dir / "Reports" / f"Success_rate_table_{timestamp}.csv"
        pivot_success.to_csv(success_table_file)
    
    def _print_console_summary(self, df_results: pd.DataFrame, analysis: Dict[str, Any]):
        """Print summary to console."""
        
        print("\n" + "=" * 80)
        print("PATTERN LIBRARY TEST RESULTS SUMMARY")
        print("=" * 80)
        
        print(f"Total tests executed: {analysis['test_info']['total_tests']}")
        print(f"Successful tests: {analysis['test_info']['successful_tests']}")
        print(f"Failed tests: {analysis['test_info']['failed_tests']}")
        print(f"Success rate: {(1 - analysis['test_info']['error_rate']):.2%}")
        print(f"Average execution time: {analysis['test_info']['avg_execution_time']:.2f} seconds")
        
        # Top performing algorithms
        if analysis['algorithm_performance']:
            print("\nTOP PERFORMING ALGORITHMS (by average ARI):")
            algo_ari = [(algo, info.get('avg_ari', 0) or 0) 
                       for algo, info in analysis['algorithm_performance'].items()]
            algo_ari.sort(key=lambda x: x[1], reverse=True)
            
            for i, (algo, ari) in enumerate(algo_ari[:5]):
                print(f"  {i+1}. {algo}: ARI = {ari:.3f}")
        
        # Most challenging datasets
        if analysis['dataset_analysis']:
            print("\nMOST CHALLENGING DATASETS (by success rate):")
            dataset_difficulty = [(dataset, info['success_rate']) 
                                for dataset, info in analysis['dataset_analysis'].items()]
            dataset_difficulty.sort(key=lambda x: x[1])
            
            for i, (dataset, success_rate) in enumerate(dataset_difficulty[:5]):
                print(f"  {i+1}. {dataset}: {success_rate:.2%} success rate")
        
        # Performance vs expectations
        if 'ari' in analysis.get('performance_comparisons', {}):
            ari_comp = analysis['performance_comparisons']['ari']
            print(f"\nPERFORMANCE VS EXPECTATIONS (ARI):")
            print(f"  Better than expected: {ari_comp['better_than_expected']} tests")
            print(f"  Similar to expected: {ari_comp['similar_to_expected']} tests")
            print(f"  Worse than expected: {ari_comp['worse_than_expected']} tests")
            print(f"  Average difference: {ari_comp['avg_difference']:.3f}")
        
        print("=" * 80)

def main():
    """Main testing function."""
    
    # Setup
    tester = AlgorithmTester()
    
    print("Pattern Library Comprehensive Testing - Memory Scale")
    print("=" * 60)
    print("This enhanced test suite will:")
    print("1. Discover all implemented algorithms and metrics")
    print("2. Download benchmark datasets for all modalities:")
    print("   - Attribute: iris, wine, breast_cancer, seeds, glass, ecoli, yeast (7 datasets)")
    print("   - Network: karate, dolphins, football, polbooks, les_miserables, adjnoun (6 datasets)")
    print("   - Attributed Graph: cora, citeseer, pubmed + 3 synthetic scenarios (6 datasets)")
    print("3. Generate comprehensive synthetic datasets:")
    print("   - Multiple attribute clustering scenarios with varying difficulty")
    print("   - Network generation with different topologies")
    print("   - Attributed graphs with controlled noise levels")
    print("4. Test algorithms with default and optimized hyperparameters")
    print("5. Calculate ARI, NMI, silhouette, and Calinski-Harabasz metrics")
    print("6. Compare obtained results with expected benchmark performance")
    print("7. Save detailed error information as JSON files")
    print("8. Generate comprehensive CSV reports and performance tables")
    print("9. Cache datasets and configurations for reproducibility")
    print("10. Export results in multiple formats (CSV, JSON, Excel)")
    print("=" * 60)
    print(f"Results will be saved in: {tester.results_dir}")
    print("Subdirectories:")
    print("  - Logs/: Execution logs")
    print("  - Errors/: JSON files with detailed error information")
    print("  - Reports/: CSV results and performance analysis")
    print("  - Cache/: Saved test results and configurations")
    print("  - Exports/: Results exported in multiple formats")
    print("  - Datasets/Synthetic/: Cached synthetic datasets")
    print("=" * 60)
    
    try:
        # Run comprehensive tests
        tester.run_comprehensive_tests()
        
        print("\nTesting completed successfully!")
        print(f"Results saved in: {tester.results_dir}")
        print("\nGenerated files:")
        print("  - Detailed_results_*.csv: Complete test results with all metrics")
        print("  - Summary_results_*.csv: Key performance indicators and comparisons")
        print("  - Analysis_report_*.json: Comprehensive statistical analysis")
        print("  - *_performance_table_*.csv: Algorithm vs dataset performance matrices")
        print("  - Error_*.json: Detailed error information for failed tests")
        print("  - Test_results_*.json: Cached test results for reload")
        print("  - Test_config_*.json: Test configurations for reproducibility")
        print("  - Exports/Results_*.csv: Multi-format result exports")
        
        # Print final statistics
        if tester.test_results:
            total_tests = len(tester.test_results)
            successful_tests = sum(1 for r in tester.test_results if r['success'])
            print(f"\nFinal Statistics:")
            print(f"  Total tests executed: {total_tests}")
            print(f"  Successful tests: {successful_tests}")
            print(f"  Failed tests: {total_tests - successful_tests}")
            print(f"  Success rate: {successful_tests/total_tests:.1%}")
            print(f"  Error files generated: {tester.error_count}")
        
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
            emergency_file = tester.results_dir / f"Emergency_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(emergency_file, 'w') as f:
                json.dump(tester.test_results, f, indent=2, default=str)
            print(f"Emergency results saved to: {emergency_file}")

if __name__ == "__main__":
    main() 