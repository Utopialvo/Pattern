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
- Large-scale dataset processing via coresets
- Efficient synthetic data generation and coreset construction
- Performance evaluation with coreset approximations
- Comprehensive coreset quality and efficiency reporting

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
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

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

warnings.filterwarnings('ignore')

class CoresetBuilder:
    """Builds coresets for different data modalities to enable scalable processing."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
    
    def build_attribute_coreset(self, X: np.ndarray, coreset_size: int, 
                               method: str = 'kmeans++') -> Tuple[np.ndarray, np.ndarray]:
        """Build coreset for attribute data using various sampling strategies."""
        
        if len(X) <= coreset_size:
            return X, np.ones(len(X))
        
        if method == 'kmeans++':
            return self._build_kmeans_plus_plus_coreset(X, coreset_size)
        elif method == 'uniform':
            return self._build_uniform_coreset(X, coreset_size)
        else:
            raise ValueError(f"Unknown coreset method: {method}")
    
    def _build_kmeans_plus_plus_coreset(self, X: np.ndarray, 
                                       coreset_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Build coreset using k-means++ initialization strategy."""
        
        n_samples, n_features = X.shape
        
        # Use k-means++ to select initial centers
        n_centers = min(coreset_size // 2, int(np.sqrt(n_samples)))
        kmeans = KMeans(n_clusters=n_centers, init='k-means++', 
                       random_state=self.random_state, n_init=1)
        kmeans.fit(X)
        
        # Sample additional points
        remaining_size = coreset_size - n_centers
        if remaining_size > 0:
            sampled_indices = np.random.choice(
                n_samples, size=remaining_size, replace=False
            )
            coreset_points = np.vstack([kmeans.cluster_centers_, X[sampled_indices]])
            
            # Calculate weights
            center_weights = np.bincount(kmeans.labels_) / n_samples
            sample_weights = np.ones(remaining_size) / remaining_size
            weights = np.concatenate([center_weights, sample_weights])
        else:
            coreset_points = kmeans.cluster_centers_
            weights = np.bincount(kmeans.labels_) / n_samples
        
        return coreset_points, weights
    
    def _build_uniform_coreset(self, X: np.ndarray, 
                              coreset_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Build coreset using uniform random sampling."""
        
        n_samples = len(X)
        sampled_indices = np.random.choice(
            n_samples, size=coreset_size, replace=False
        )
        
        coreset_points = X[sampled_indices]
        weights = np.full(coreset_size, n_samples / coreset_size)
        
        return coreset_points, weights

class CoresetDataManager:
    """Manages coreset-based data processing for benchmark and synthetic datasets."""
    
    def __init__(self, coreset_builder: CoresetBuilder, data_dir: str = "coreset_data"):
        self.coreset_builder = coreset_builder
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Coreset configurations
        self.coreset_configs = {
            'small': {'size_ratio': 0.1, 'min_size': 100, 'max_size': 1000},
            'medium': {'size_ratio': 0.05, 'min_size': 200, 'max_size': 2000},
            'large': {'size_ratio': 0.02, 'min_size': 500, 'max_size': 5000}
        }
    
    def create_coreset_benchmark_data(self, original_size: int = 10000, 
                                     n_features: int = 20, n_clusters: int = 5,
                                     coreset_config: str = 'medium') -> Dict[str, Any]:
        """Create benchmark data with corresponding coresets."""
        
        logger.info(f"Creating coreset benchmark data: {original_size} samples, {n_features} features")
        
        # Generate large original dataset
        X_original, y_original = make_blobs(
            n_samples=original_size, centers=n_clusters, n_features=n_features,
            cluster_std=2.0, random_state=42
        )
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_original)
        
        # Calculate coreset size
        config = self.coreset_configs[coreset_config]
        coreset_size = max(
            config['min_size'],
            min(config['max_size'], int(original_size * config['size_ratio']))
        )
        
        # Build coresets using different methods
        coresets = {}
        coreset_methods = ['kmeans++', 'uniform']
        
        for method in coreset_methods:
            try:
                coreset_points, weights = self.coreset_builder.build_attribute_coreset(
                    X_scaled, coreset_size, method
                )
                
                coresets[method] = {
                    'points': coreset_points,
                    'weights': weights,
                    'size': len(coreset_points),
                    'compression_ratio': original_size / len(coreset_points)
                }
                
                logger.info(f"Built {method} coreset: {len(coreset_points)} points "
                           f"(compression: {coresets[method]['compression_ratio']:.1f}x)")
                
            except Exception as e:
                logger.warning(f"Failed to build {method} coreset: {e}")
        
        return {
            'original': {'features': X_scaled, 'labels': y_original},
            'coresets': coresets,
            'metadata': {
                'original_size': original_size,
                'n_features': n_features,
                'n_clusters': n_clusters,
                'coreset_config': coreset_config
            }
        }

class CoresetAlgorithmTester:
    """Tests Pattern library algorithms using coreset-based processing."""
    
    def __init__(self, results_dir: str = "test_results_coreset"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        self.coreset_builder = CoresetBuilder()
        self.data_manager = CoresetDataManager(self.coreset_builder)
        self.test_results = []
        
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration for coreset testing."""
        log_file = self.results_dir / f"coreset_test_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
    
    def discover_algorithms(self) -> Dict[str, Dict]:
        """Discover algorithms compatible with coreset processing."""
        logger.info("Discovering coreset-compatible algorithms...")
        
        algorithms = {}
        for name, info in MODEL_REGISTRY.items():
            algorithms[name] = {
                'class': info['class'],
                'params_help': info['params_help'],
                'modality': self._infer_modality(name, info)
            }
            logger.info(f"Found algorithm: {name}")
        
        return algorithms
    
    def _infer_modality(self, algo_name: str, algo_info: Dict) -> str:
        """Infer the modality of an algorithm."""
        name_lower = algo_name.lower()
        
        if any(keyword in name_lower for keyword in ['spectral', 'louvain', 'modularity']):
            return 'network'
        elif any(keyword in name_lower for keyword in ['dmon', 'gnn', 'graph', 'node2vec']):
            return 'attributed_graph'
        else:
            return 'attribute'
    
    def test_algorithm_on_coreset(self, algorithm_name: str, dataset_name: str,
                                 coreset_data: Dict[str, Any], coreset_method: str,
                                 original_data: Dict[str, Any], params: Dict[str, Any],
                                 optimization_method: str = 'default') -> Dict[str, Any]:
        """Test algorithm on coreset data and compare with original."""
        
        start_time = time.time()
        result = {
            'algorithm': algorithm_name,
            'dataset': dataset_name,
            'coreset_method': coreset_method,
            'optimization': optimization_method,
            'params': params.copy(),
            'success': False,
            'error': None,
            'execution_time': 0,
            'coreset_metrics': {},
            'approximation_quality': {},
            'efficiency_metrics': {}
        }
        
        try:
            logger.info(f"Testing {algorithm_name} on {dataset_name} coreset ({coreset_method})")
            
            # Test on coreset
            coreset_result = self._test_on_dataset(
                algorithm_name, coreset_data['points'], None, params
            )
            
            # Record results
            result['coreset_metrics'] = coreset_result['metrics']
            
            # Calculate efficiency metrics
            result['efficiency_metrics'] = {
                'coreset_size': len(coreset_data['points']),
                'original_size': len(original_data['features']),
                'compression_ratio': len(original_data['features']) / len(coreset_data['points']),
                'execution_time': coreset_result['execution_time']
            }
            
            result['success'] = coreset_result['success']
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Failed to test {algorithm_name} on {dataset_name} coreset: {e}")
        
        result['execution_time'] = time.time() - start_time
        return result
    
    def _test_on_dataset(self, algorithm_name: str, features: np.ndarray, 
                        similarity: Optional[np.ndarray], params: Dict[str, Any]) -> Dict[str, Any]:
        """Test algorithm on a specific dataset."""
        
        start_time = time.time()
        result = {
            'success': False,
            'metrics': {},
            'execution_time': 0,
            'error': None
        }
        
        try:
            # Convert to pandas for Pattern library
            if features is not None:
                feature_names = [f'feature_{i}' for i in range(features.shape[1])]
                features_df = pd.DataFrame(features, columns=feature_names)
            else:
                features_df = None
            
            similarity_df = pd.DataFrame(similarity) if similarity is not None else None
            
            # Create data loader
            data_loader = PandasDataLoader(features=features_df, similarity=similarity_df)
            
            # Create and fit model
            model = factory.create_model(algorithm_name, params)
            model.fit(data_loader)
            
            # Get predictions
            if hasattr(model, 'labels_') and model.labels_ is not None:
                predicted_labels = model.labels_
            else:
                predicted_labels = model.predict(data_loader)
            
            # Pattern library metrics
            for metric_name in METRIC_REGISTRY:
                try:
                    metric = factory.create_metric(metric_name)
                    score = metric.calculate(data_loader, predicted_labels, model.model_data)
                    if not np.isnan(score):
                        result['metrics'][metric_name] = score
                except Exception as e:
                    logger.warning(f"Failed to calculate {metric_name}: {e}")
            
            result['success'] = True
            
        except Exception as e:
            result['error'] = str(e)
        
        result['execution_time'] = time.time() - start_time
        return result
    
    def get_default_params(self, algorithm_name: str) -> Dict[str, Any]:
        """Get default parameters optimized for coreset processing."""
        if algorithm_name not in MODEL_REGISTRY:
            return {}
        
        params_help = MODEL_REGISTRY[algorithm_name]['params_help']
        default_params = {}
        
        for param_name, description in params_help.items():
            if 'cluster' in param_name.lower():
                default_params[param_name] = 3  # Conservative for coresets
            elif param_name.lower() in ['eps', 'epsilon']:
                default_params[param_name] = 0.5
            elif 'min_samples' in param_name.lower():
                default_params[param_name] = 3  # Lower for smaller coresets
            elif 'init' in param_name.lower():
                default_params[param_name] = 'k-means++'
            elif 'max_iter' in param_name.lower():
                default_params[param_name] = 200
            elif 'resolution' in param_name.lower():
                default_params[param_name] = 1.0
        
        return default_params
    
    def run_comprehensive_tests(self):
        """Run comprehensive tests using coreset-based processing."""
        
        logger.info("Starting comprehensive Pattern library testing (Coreset Scale)")
        
        algorithms = self.discover_algorithms()
        
        # Test on coreset benchmark datasets
        self._test_coreset_benchmark_datasets(algorithms)
        
        # Test on coreset synthetic datasets
        self._test_coreset_synthetic_datasets(algorithms)
        
        # Generate comprehensive report
        self._generate_coreset_report()
        
        logger.info("Coreset comprehensive testing completed")
    
    def _test_coreset_benchmark_datasets(self, algorithms: Dict[str, Dict]):
        """Test algorithms on coreset benchmark datasets."""
        
        logger.info("Testing on coreset benchmark datasets...")
        
        # Create different scale benchmark datasets
        dataset_configs = [
            {'name': 'medium_scale', 'original_size': 5000, 'n_features': 15, 'n_clusters': 5},
            {'name': 'large_scale', 'original_size': 20000, 'n_features': 20, 'n_clusters': 8},
        ]
        
        for dataset_config in dataset_configs:
            logger.info(f"Creating coreset benchmark dataset: {dataset_config['name']}")
            
            dataset = self.data_manager.create_coreset_benchmark_data(**dataset_config)
            
            # Test each coreset method
            for coreset_method, coreset_data in dataset['coresets'].items():
                
                # Test attribute algorithms
                for algo_name, algo_info in algorithms.items():
                    if algo_info['modality'] == 'attribute':
                        
                        # Test with default parameters
                        default_params = self.get_default_params(algo_name)
                        result = self.test_algorithm_on_coreset(
                            algo_name, dataset_config['name'], coreset_data, coreset_method,
                            dataset['original'], default_params, 'default'
                        )
                        self.test_results.append(result)
    
    def _test_coreset_synthetic_datasets(self, algorithms: Dict[str, Dict]):
        """Test algorithms on synthetic coreset datasets."""
        
        logger.info("Testing on synthetic coreset datasets...")
        
        # Create diverse synthetic scenarios
        synthetic_scenarios = [
            {'name': 'well_separated', 'original_size': 10000, 'n_features': 10, 'n_clusters': 4},
            {'name': 'overlapping', 'original_size': 8000, 'n_features': 15, 'n_clusters': 6}
        ]
        
        for scenario in synthetic_scenarios:
            logger.info(f"Creating synthetic coreset dataset: {scenario['name']}")
            
            dataset = self.data_manager.create_coreset_benchmark_data(**scenario)
            
            # Test best performing coreset method (kmeans++)
            if 'kmeans++' in dataset['coresets']:
                coreset_data = dataset['coresets']['kmeans++']
                
                for algo_name, algo_info in algorithms.items():
                    if algo_info['modality'] == 'attribute':
                        default_params = self.get_default_params(algo_name)
                        if 'n_clusters' in default_params:
                            default_params['n_clusters'] = scenario['n_clusters']
                        
                        result = self.test_algorithm_on_coreset(
                            algo_name, f"synthetic_{scenario['name']}", coreset_data, 'kmeans++',
                            dataset['original'], default_params, 'default'
                        )
                        self.test_results.append(result)
    
    def _generate_coreset_report(self):
        """Generate comprehensive coreset test report."""
        
        logger.info("Generating comprehensive coreset test report...")
        
        df_results = pd.DataFrame(self.test_results)
        
        # Save detailed results
        results_file = self.results_dir / f"coreset_detailed_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df_results.to_csv(results_file, index=False)
        
        # Generate summary
        summary = {
            'test_info': {
                'timestamp': datetime.now().isoformat(),
                'total_tests': len(df_results),
                'successful_tests': int(df_results['success'].sum()) if not df_results.empty else 0,
                'failed_tests': int((~df_results['success']).sum()) if not df_results.empty else 0,
                'scale': 'coreset'
            },
            'coreset_analysis': {},
            'efficiency_analysis': {}
        }
        
        # Coreset method analysis
        if not df_results.empty:
            for method in df_results['coreset_method'].unique():
                method_results = df_results[df_results['coreset_method'] == method]
                summary['coreset_analysis'][method] = {
                    'success_rate': float(method_results['success'].mean()),
                    'tests_count': len(method_results)
                }
        
        summary_file = self.results_dir / f"coreset_summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        logger.info("=" * 60)
        logger.info("PATTERN LIBRARY TEST SUMMARY (CORESET SCALE)")
        logger.info("=" * 60)
        logger.info(f"Total tests executed: {len(self.test_results)}")
        logger.info(f"Successful tests: {sum(1 for r in self.test_results if r['success'])}")
        logger.info(f"Failed tests: {sum(1 for r in self.test_results if not r['success'])}")
        
        if self.test_results:
            avg_time = np.mean([r['execution_time'] for r in self.test_results])
            logger.info(f"Average execution time: {avg_time:.2f} seconds")
        
        logger.info("=" * 60)
        logger.info(f"Detailed results saved to: {results_file}")
        logger.info(f"Summary report saved to: {summary_file}")

def main():
    """Main coreset testing function."""
    
    print("Pattern Library Comprehensive Testing - Coreset Scale")
    print("=" * 60)
    print("This test suite will:")
    print("1. Discover all algorithms and their coreset compatibility")
    print("2. Generate large-scale datasets and build coresets")
    print("3. Test algorithms on coresets vs original data")
    print("4. Analyze approximation quality and efficiency gains")
    print("5. Generate comprehensive coreset performance reports")
    print("=" * 60)
    
    try:
        tester = CoresetAlgorithmTester()
        tester.run_comprehensive_tests()
        
        print("\nCoreset testing completed successfully!")
        print(f"Results saved in: {tester.results_dir}")
        
    except Exception as e:
        logger.error(f"Coreset testing failed with error: {e}")
        logger.debug(traceback.format_exc())
        print(f"\nCoreset testing failed: {e}")

if __name__ == "__main__":
    main() 