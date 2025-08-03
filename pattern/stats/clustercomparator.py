# Файл: stats/clustercomparator.py

import numpy as np
import os
import joblib
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from scipy.sparse import issparse, csr_matrix
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import normalize
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import networkx as nx
import pandas as pd
import time
from datetime import datetime
from abc import ABC, abstractmethod
from itertools import combinations
from functools import lru_cache
from typing import List, Dict, Tuple, Union, Any
import threading

class BaseComparator(ABC):
    def __init__(self, labels_list: List[np.ndarray], output_dir: str = "reports", jaccard_threshold: float = 0.0):
        """Base class for comparing multiple clustering solutions."""
        self.labels_list = labels_list
        self.n_clusterings = len(labels_list)
        self.jaccard_threshold = jaccard_threshold
        self._validate_labels()
        
        # Create unique timestamped output directory
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir) / self.timestamp
        self.cache_dir = self.output_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "centroids").mkdir(exist_ok=True)
        (self.cache_dir / "mappings").mkdir(exist_ok=True)

        # Thread-safe initialization
        self.centroids_cache = {}
        self.normalization_factors = None
        self.global_metrics = None
        self.graph_metrics = None
        self.full_report = None
        self._cache_lock = threading.Lock()

    def _validate_labels(self) -> None:
        """Validates labels input."""
        n_samples = len(self.labels_list[0])
        for i, labels in enumerate(self.labels_list):
            if len(labels) != n_samples:
                raise ValueError(f"Labels[{i}] length {len(labels)} doesn't match first clustering {n_samples}")
            if np.any(labels < -1):
                raise ValueError(f"Labels[{i}] contain negative values < -1")
            if not isinstance(labels, np.ndarray):
                raise TypeError(f"Labels[{i}] must be numpy array")

    @abstractmethod
    def get_centroids(self, i: int, centroid_type: str) -> np.ndarray:
        """Retrieves centroids from cache or computes them."""
        pass

    @abstractmethod
    def compute_global_metrics(self) -> Dict[str, Any]:
        """Computes global metrics for all clustering solutions."""
        pass

    def _get_normalization_factors(self) -> Dict[str, float]:
        """Computes normalization factors using median of pairwise distances."""
        if self.normalization_factors is not None:
            return self.normalization_factors

        cache_path = self.cache_dir / "normalization_factors.pkl"
        if cache_path.exists():
            with self._cache_lock:
                self.normalization_factors = joblib.load(cache_path)
            return self.normalization_factors

        centroid_types = ['X', 'A', 'X_w', 'A_w']
        self.normalization_factors = {}

        # Precompute centroids and cluster sizes
        all_centroids = {ct: [] for ct in centroid_types}
        cluster_sizes = {ct: [] for ct in centroid_types}
        
        for i in range(self.n_clusterings):
            for ct in centroid_types:
                centroids = self.get_centroids(i, ct)
                if len(centroids) > 0:
                    all_centroids[ct].append(centroids)
                    labels = self.labels_list[i]
                    sizes = [np.sum(labels == k) for k in np.unique(labels) if k != -1]
                    cluster_sizes[ct].append(sizes)

        # Compute in parallel with limited workers
        max_workers = min(4, os.cpu_count() or 1)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for ct in centroid_types:
                if not all_centroids[ct]:
                    self.normalization_factors[ct] = 1.0
                    continue
                futures[executor.submit(
                    self._compute_normalization_for_type,
                    all_centroids[ct],
                    cluster_sizes[ct]
                )] = ct

            for future in as_completed(futures):
                ct = futures[future]
                try:
                    self.normalization_factors[ct] = future.result()
                except Exception as e:
                    warnings.warn(f"Could not compute normalization for {ct}: {str(e)}")
                    self.normalization_factors[ct] = 1.0

        with self._cache_lock:
            joblib.dump(self.normalization_factors, cache_path)
        return self.normalization_factors

    def _compute_normalization_for_type(self, centroids_list: List[np.ndarray], sizes_list: List[List[int]]) -> float:
        """Helper method to compute normalization for a centroid type."""
        weighted_dists = []
        n_clusterings = len(centroids_list)
        
        for i in range(n_clusterings):
            for j in range(i + 1, n_clusterings):
                centroids_i = centroids_list[i]
                centroids_j = centroids_list[j]
                sizes_i = sizes_list[i]
                sizes_j = sizes_list[j]
                
                dists = cdist(centroids_i, centroids_j)
                weights = np.outer(sizes_i, sizes_j)
                weighted_avg = np.average(dists, weights=weights)
                weighted_dists.append(weighted_avg)
        
        return np.median(weighted_dists) + 1e-10 if weighted_dists else 1.0

    @lru_cache(maxsize=None)
    def get_cluster_mapping(self, i: int, j: int) -> List[Dict]:
        """Finds optimal cluster mapping between two clusterings."""
        cache_path = self.cache_dir / "mappings" / f"mapping_{i}_{j}.pkl"
        if cache_path.exists():
            with self._cache_lock:
                return joblib.load(cache_path)

        labels_i = self.labels_list[i]
        labels_j = self.labels_list[j]
        unique_i = np.setdiff1d(np.unique(labels_i), [-1])
        unique_j = np.setdiff1d(np.unique(labels_j), [-1])

        # Compute Jaccard similarity matrix
        similarity_matrix = np.zeros((len(unique_i), len(unique_j)))
        masks_i = [labels_i == k for k in unique_i]
        masks_j = [labels_j == k for k in unique_j]
        
        for idx_i, mask_i in enumerate(masks_i):
            for idx_j, mask_j in enumerate(masks_j):
                intersection = np.sum(mask_i & mask_j)
                union = np.sum(mask_i | mask_j)
                similarity_matrix[idx_i, idx_j] = intersection / union if union > 0 else 0

        # Find optimal assignment
        row_idx, col_idx = linear_sum_assignment(-similarity_matrix)
        mappings = [
            {
                'cluster_i': int(unique_i[r]),
                'cluster_j': int(unique_j[c]),
                'similarity': similarity_matrix[r, c]
            }
            for r, c in zip(row_idx, col_idx)
            if similarity_matrix[r, c] >= self.jaccard_threshold
        ]

        with self._cache_lock:
            joblib.dump(mappings, cache_path)
        return mappings

    def compute_cluster_distances(self, i: int, j: int, cluster_i: int, cluster_j: int) -> Dict[str, float]:
        """Computes distances between specific clusters."""
        distances = {}
        norm_factors = self._get_normalization_factors()

        for c_type in ['X', 'A', 'X_w', 'A_w']:
            centroids_i = self.get_centroids(i, c_type)
            centroids_j = self.get_centroids(j, c_type)

            if len(centroids_i) == 0 or len(centroids_j) == 0:
                distances[c_type] = float('inf')
                continue

            try:
                unique_i = np.setdiff1d(np.unique(self.labels_list[i]), [-1])
                unique_j = np.setdiff1d(np.unique(self.labels_list[j]), [-1])
                
                cluster_to_index_i = {k: idx for idx, k in enumerate(unique_i)}
                cluster_to_index_j = {k: idx for idx, k in enumerate(unique_j)}
                
                idx_i = cluster_to_index_i[cluster_i]
                idx_j = cluster_to_index_j[cluster_j]
                raw_dist = np.linalg.norm(centroids_i[idx_i] - centroids_j[idx_j])
                distances[c_type] = raw_dist / norm_factors[c_type]
            except Exception as e:
                distances[c_type] = float('inf')
                warnings.warn(f"Distance computation failed for {c_type}: {str(e)}")

        return {
            'feature_distance': distances.get('X', float('inf')),
            'topology_distance': distances.get('A', float('inf')),
            'feature_weighted_distance': distances.get('X_w', float('inf')),
            'topology_weighted_distance': distances.get('A_w', float('inf'))
        }

    def generate_pairwise_report(self, i: int, j: int) -> Dict[str, Any]:
        """Generates detailed comparison report for two clusterings."""
        ari = adjusted_rand_score(self.labels_list[i], self.labels_list[j])
        nmi = normalized_mutual_info_score(self.labels_list[i], self.labels_list[j])
        mappings = self.get_cluster_mapping(i, j)

        # Compute distances for mapped clusters
        cluster_distances = []
        for mapping in mappings:
            try:
                dist_data = self.compute_cluster_distances(
                    i, j, mapping['cluster_i'], mapping['cluster_j']
                )
                cluster_distances.append({
                    'cluster_i': mapping['cluster_i'],
                    'cluster_j': mapping['cluster_j'],
                    'similarity': mapping['similarity'],
                    'distances': dist_data
                })
            except Exception as e:
                warnings.warn(f"Distance computation failed for clusters {mapping['cluster_i']} and {mapping['cluster_j']}: {str(e)}")

        # Identify unmapped clusters
        unique_i = np.setdiff1d(np.unique(self.labels_list[i]), [-1])
        unique_j = np.setdiff1d(np.unique(self.labels_list[j]), [-1])
        mapped_i = {m['cluster_i'] for m in mappings}
        mapped_j = {m['cluster_j'] for m in mappings}
        unmapped_i = [int(k) for k in unique_i if k not in mapped_i]
        unmapped_j = [int(k) for k in unique_j if k not in mapped_j]

        return {
            'clustering_i': i,
            'clustering_j': j,
            'global_metrics': {'ARI': ari, 'NMI': nmi},
            'cluster_mappings': mappings,
            'cluster_distances': cluster_distances,
            'unmapped_clusters': {
                f'clustering_{i}': unmapped_i,
                f'clustering_{j}': unmapped_j
            }
        }

    def generate_full_report(self) -> Dict[str, Any]:
        """Generates comprehensive comparison report for all clustering pairs."""
        start_time = time.time()
        print(f"Starting full report generation at {self.timestamp}")

        # Compute global metrics
        global_metrics = self.compute_global_metrics()

        report = {
            'timestamp': self.timestamp,
            'parameters': {'jaccard_threshold': self.jaccard_threshold},
            'global_comparison': {
                'ari_matrix': np.zeros((self.n_clusterings, self.n_clusterings)),
                'nmi_matrix': np.zeros((self.n_clusterings, self.n_clusterings)),
                'global_metrics': global_metrics
            },
            'pairwise_reports': {}
        }

        # Compute similarity matrices
        ari_matrix = np.zeros((self.n_clusterings, self.n_clusterings))
        nmi_matrix = np.zeros((self.n_clusterings, self.n_clusterings))
        np.fill_diagonal(ari_matrix, 1.0)
        np.fill_diagonal(nmi_matrix, 1.0)
        
        # Compute pairwise similarities
        pairs = list(combinations(range(self.n_clusterings), 2))
        max_workers = min(4, os.cpu_count() or 1)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit ARI and NMI together for each pair
            future_to_pair = {}
            for i, j in pairs:
                future_ari = executor.submit(adjusted_rand_score, self.labels_list[i], self.labels_list[j])
                future_nmi = executor.submit(normalized_mutual_info_score, self.labels_list[i], self.labels_list[j])
                future_to_pair[future_ari] = ('ari', i, j)
                future_to_pair[future_nmi] = ('nmi', i, j)
            
            # Process results
            for future in as_completed(future_to_pair):
                metric_type, i, j = future_to_pair[future]
                try:
                    result = future.result()
                    if metric_type == 'ari':
                        ari_matrix[i, j] = result
                        ari_matrix[j, i] = result
                    else:
                        nmi_matrix[i, j] = result
                        nmi_matrix[j, i] = result
                except Exception as e:
                    print(f"Error computing {metric_type.upper()} for ({i},{j}): {e}")

        report['global_comparison']['ari_matrix'] = ari_matrix
        report['global_comparison']['nmi_matrix'] = nmi_matrix

        # Generate pairwise reports
        pairwise_reports = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {}
            for i, j in pairs:
                future = executor.submit(self.generate_pairwise_report, i, j)
                future_to_index[future] = (i, j)
            
            for future in as_completed(future_to_index):
                i, j = future_to_index[future]
                try:
                    report_key = f"clustering_{i}_vs_{j}"
                    pairwise_reports[report_key] = future.result()
                except Exception as e:
                    print(f"Error generating report for ({i},{j}): {e}")
                    pairwise_reports[report_key] = {"error": str(e)}

        report['pairwise_reports'] = pairwise_reports
        self.full_report = report
        elapsed = time.time() - start_time
        print(f"Report generated in {elapsed:.2f} seconds. Output directory: {self.output_dir}")
        return report

    def export_to_dataframes(self) -> Dict[str, pd.DataFrame]:
        """Exports full report to a collection of pandas DataFrames."""
        if self.full_report is None:
            self.generate_full_report()

        dfs = {}
        report = self.full_report
        df_dir = self.output_dir / "dataframes"
        df_dir.mkdir(exist_ok=True)

        # Global metrics per clustering
        global_data = []
        global_metrics = report['global_comparison']['global_metrics']
        
        for i in range(self.n_clusterings):
            sil = global_metrics['silhouette'][i]
            gdist = global_metrics['global_distance'][i]
            graph_metrics = global_metrics['graph_metrics']
            
            global_data.append({
                'clustering_index': i,
                'silhouette_features': sil['features'],
                'silhouette_adjacency': sil['adjacency'],
                'global_dist_X': gdist['X'],
                'global_dist_A': gdist['A'],
                'global_dist_X_w': gdist['X_w'],
                'global_dist_A_w': gdist['A_w'],
                'modularity': graph_metrics['modularity'][i],
                'conductance': graph_metrics['conductance'][i],
                'coverage': graph_metrics['coverage'][i],
                'performance': graph_metrics['performance'][i]
            })
            
        dfs['global_metrics'] = pd.DataFrame(global_data)

        # Similarity matrices
        dfs['similarity_matrices'] = {
            'ARI': pd.DataFrame(
                report['global_comparison']['ari_matrix'],
                index=range(self.n_clusterings),
                columns=range(self.n_clusterings)
            ),
            'NMI': pd.DataFrame(
                report['global_comparison']['nmi_matrix'],
                index=range(self.n_clusterings),
                columns=range(self.n_clusterings)
            )
        }

        # Pairwise mappings and distances
        mappings_data = []
        distances_data = []
        unmapped_data = []

        for key, pairwise_report in report['pairwise_reports'].items():
            # Skip error reports
            if "error" in pairwise_report:
                continue
                
            i = pairwise_report['clustering_i']
            j = pairwise_report['clustering_j']

            # Cluster mappings
            for mapping in pairwise_report['cluster_mappings']:
                mappings_data.append({
                    'clustering_pair': key,
                    'clustering_i': i,
                    'clustering_j': j,
                    'cluster_i': mapping['cluster_i'],
                    'cluster_j': mapping['cluster_j'],
                    'similarity': mapping['similarity']
                })

            # Cluster distances
            for dist in pairwise_report['cluster_distances']:
                dist_values = dist['distances']
                distances_data.append({
                    'clustering_pair': key,
                    'clustering_i': i,
                    'clustering_j': j,
                    'cluster_i': dist['cluster_i'],
                    'cluster_j': dist['cluster_j'],
                    'feature_distance': dist_values['feature_distance'],
                    'topology_distance': dist_values['topology_distance'],
                    'feature_weighted_distance': dist_values['feature_weighted_distance'],
                    'topology_weighted_distance': dist_values['topology_weighted_distance']
                })

            # Unmapped clusters
            unmapped = pairwise_report['unmapped_clusters']
            for cluster in unmapped[f'clustering_{i}']:
                unmapped_data.append({
                    'clustering_pair': key,
                    'clustering_index': i,
                    'cluster_id': cluster
                })
            for cluster in unmapped[f'clustering_{j}']:
                unmapped_data.append({
                    'clustering_pair': key,
                    'clustering_index': j,
                    'cluster_id': cluster
                })

        dfs['pairwise_mappings'] = pd.DataFrame(mappings_data)
        dfs['cluster_distances'] = pd.DataFrame(distances_data)
        dfs['unmapped_clusters'] = pd.DataFrame(unmapped_data)

        # Save all DataFrames to CSV
        for name, df in dfs.items():
            if isinstance(df, dict):
                for matrix_name, matrix_df in df.items():
                    path = df_dir / f"{name}_{matrix_name}.csv"
                    matrix_df.to_csv(path, index=True)
            else:
                path = df_dir / f"{name}.csv"
                df.to_csv(path, index=False)

        print(f"DataFrames saved to: {df_dir}")
        return dfs


class SingleDatasetComparator(BaseComparator):
    def __init__(self, X: np.ndarray, A: Union[csr_matrix, np.ndarray], 
                 labels_list: List[np.ndarray], **kwargs):
        """Compares clusterings on the same dataset."""
        super().__init__(labels_list, **kwargs)
        self.X = X
        self.A = A if issparse(A) else csr_matrix(A)
        
        # Validate input dimensions
        n_samples = len(labels_list[0])
        if X.shape[0] != n_samples:
            raise ValueError(f"Feature matrix shape {X.shape} doesn't match labels {n_samples}")
        if self.A.shape[0] != n_samples or self.A.shape[1] != n_samples:
            raise ValueError(f"Adjacency matrix shape {self.A.shape} doesn't match labels {n_samples}")
            
        # Compute global reference points
        self.global_X = np.mean(X, axis=0)
        self.global_A = np.array(self.A.mean(axis=0)).flatten()
        
        # Cache graph and total edge weight
        self.G = nx.from_scipy_sparse_array(self.A)
        self.total_edge_weight = self.A.sum() / 2

    def get_centroids(self, i: int, centroid_type: str) -> np.ndarray:
        """Retrieves centroids from cache or computes them."""
        cache_key = (i, centroid_type)
        if cache_key in self.centroids_cache:
            return self.centroids_cache[cache_key]

        cache_path = self.cache_dir / "centroids" / f"clustering_{i}_{centroid_type}.npy"
        if cache_path.exists():
            with self._cache_lock:
                centroids = np.load(cache_path)
        else:
            centroids = self._compute_centroids(self.labels_list[i], centroid_type)
            with self._cache_lock:
                np.save(cache_path, centroids)

        self.centroids_cache[cache_key] = centroids
        return centroids

    def _compute_centroids(self, labels: np.ndarray, centroid_type: str) -> np.ndarray:
        """Computes centroids for a clustering solution."""
        unique_labels = np.unique(labels)
        centroids = []

        for k in unique_labels:
            if k == -1:  # Skip noise
                continue

            mask = (labels == k)
            cluster_size = np.sum(mask)

            if cluster_size == 0:  # Skip empty clusters
                continue

            try:
                if centroid_type == 'X':
                    centroids.append(np.mean(self.X[mask], axis=0))

                elif centroid_type == 'A':
                    cluster_adj = self.A[mask].mean(axis=0)
                    if issparse(cluster_adj):
                        cluster_adj = cluster_adj.toarray().flatten()
                    centroids.append(cluster_adj)

                elif centroid_type == 'X_w':
                    weights = np.array(self.A[mask].sum(axis=1)).flatten() + 1e-10
                    centroids.append(np.average(self.X[mask], axis=0, weights=weights))

                elif centroid_type == 'A_w':
                    weights = np.array(self.A[mask].sum(axis=1)).flatten() + 1e-10
                    cluster_adj = self.A[mask].multiply(weights[:, np.newaxis]).mean(axis=0)
                    if issparse(cluster_adj):
                        cluster_adj = cluster_adj.toarray().flatten()
                    centroids.append(cluster_adj)

            except Exception as e:
                warnings.warn(f"Failed to compute {centroid_type} centroid for cluster {k}: {str(e)}")

        return np.array(centroids) if centroids else np.empty((0, self.X.shape[1]))
    
    def compute_global_metrics(self) -> Dict[str, Any]:
        """Computes global metrics for all clustering solutions."""
        if self.global_metrics is not None:
            return self.global_metrics

        # Compute in parallel
        max_workers = min(4, os.cpu_count() or 1)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._compute_metrics_for_clustering, i): i 
                      for i in range(self.n_clusterings)}
            
            results = [None] * self.n_clusterings
            for future in as_completed(futures):
                i = futures[future]
                try:
                    results[i] = future.result()
                except Exception as e:
                    print(f"Error computing metrics for clustering {i}: {e}")
                    # Return defaults on error
                    results[i] = (
                        {'features': -1, 'adjacency': -1},
                        {'X': float('inf'), 'A': float('inf'), 
                         'X_w': float('inf'), 'A_w': float('inf')}
                    )

        # Unpack results
        silhouette_metrics = [r[0] for r in results]
        global_dist_metrics = [r[1] for r in results]
        graph_metrics = self._compute_graph_metrics()

        self.global_metrics = {
            'silhouette': silhouette_metrics,
            'global_distance': global_dist_metrics,
            'graph_metrics': graph_metrics
        }
        return self.global_metrics

    def _compute_metrics_for_clustering(self, i: int) -> Tuple[Dict, Dict]:
        """Compute metrics for a single clustering solution."""
        labels = self.labels_list[i]
        
        # Silhouette scores
        try:
            sil_x = silhouette_score(self.X, labels)
        except:
            sil_x = -1
            
        try:
            D = 1 - normalize(self.A, norm='l1', axis=1)
            sil_a = silhouette_score(D, labels, metric='precomputed')
        except:
            sil_a = -1
            
        # Global distances
        dist_x = self._compute_global_distance(i, 'X')
        dist_a = self._compute_global_distance(i, 'A')
        dist_xw = self._compute_global_distance(i, 'X_w')
        dist_aw = self._compute_global_distance(i, 'A_w')
        
        return (
            {'features': sil_x, 'adjacency': sil_a},
            {'X': dist_x, 'A': dist_a, 'X_w': dist_xw, 'A_w': dist_aw}
        )

    def _compute_graph_metrics(self) -> Dict[str, List]:
        """Computes graph-specific metrics for all clusterings."""
        if self.graph_metrics is not None:
            return self.graph_metrics

        metrics = {
            'modularity': [],
            'conductance': [],
            'coverage': [],
            'performance': []
        }

        # Compute in parallel
        max_workers = min(4, os.cpu_count() or 1)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._compute_graph_metrics_for_labels, labels): i 
                      for i, labels in enumerate(self.labels_list)}
            
            results = [None] * self.n_clusterings
            for future in as_completed(futures):
                i = futures[future]
                try:
                    results[i] = future.result()
                except Exception as e:
                    print(f"Error computing graph metrics for clustering {i}: {e}")
                    results[i] = (-1, float('inf'), 0, np.nan)

        # Unpack results
        for i in range(self.n_clusterings):
            mod, cond, cov, perf = results[i]
            metrics['modularity'].append(mod)
            metrics['conductance'].append(cond)
            metrics['coverage'].append(cov)
            metrics['performance'].append(perf)

        self.graph_metrics = metrics
        return metrics

    def _compute_graph_metrics_for_labels(self, labels: np.ndarray) -> Tuple[float, float, float, float]:
        """Computes graph metrics for a single clustering solution."""
        # Modularity
        communities = self._labels_to_communities(labels)
        modularity = nx.algorithms.community.modularity(self.G, communities) if communities else -1

        # Conductance and Coverage
        conductance, coverage = self._compute_conductance_and_coverage(labels)

        # Performance (only for binary graphs)
        performance = self._compute_performance(labels) if self._is_binary() else np.nan

        return modularity, conductance, coverage, performance

    def _is_binary(self) -> bool:
        """Check if graph is binary (weights are 0 or 1)."""
        return np.all(np.isin(self.A.data, [0, 1])) if issparse(self.A) else np.all(np.isin(self.A, [0, 1]))

    def _labels_to_communities(self, labels: np.ndarray) -> List[List[int]]:
        """Converts labels to communities, excluding noise."""
        communities = {}
        for i, label in enumerate(labels):
            if label == -1:
                continue
            communities.setdefault(label, []).append(i)
        return list(communities.values())

    def _compute_conductance_and_coverage(self, labels: np.ndarray) -> Tuple[float, float]:
        """Computes average conductance and coverage for clustering."""
        unique_labels = np.setdiff1d(np.unique(labels), [-1])
        conductances = []
        total_internal_edges = 0.0

        for k in unique_labels:
            mask = (labels == k)
            cluster_size = np.sum(mask)

            if cluster_size == 0 or cluster_size == len(labels):
                continue

            # Compute cut and volume
            cut_size = self.A[mask][:, ~mask].sum()
            volume = self.A[mask].sum()
            internal_edges = self.A[mask][:, mask].sum() / 2  # Undirected
            total_internal_edges += internal_edges

            if volume > 0:
                conductances.append(cut_size / volume)

        avg_conductance = np.mean(conductances) if conductances else float('inf')
        coverage = total_internal_edges / self.total_edge_weight if self.total_edge_weight > 0 else 0
        return avg_conductance, coverage

    def _compute_performance(self, labels: np.ndarray) -> float:
        """Computes performance metric for binary graphs."""
        unique_labels = np.setdiff1d(np.unique(labels), [-1])
        n_samples = len(labels)
        total_possible_edges = n_samples * (n_samples - 1) / 2
        non_edges_inside = 0
        total_internal_edges = 0

        for k in unique_labels:
            mask = (labels == k)
            cluster_size = np.sum(mask)

            if cluster_size < 2:
                continue

            # Possible edges inside cluster
            possible_edges = cluster_size * (cluster_size - 1) / 2

            # Actual edges inside cluster
            internal_edges = self.A[mask][:, mask].sum() / 2  # Undirected
            total_internal_edges += internal_edges
            non_edges_inside += (possible_edges - internal_edges)

        cut_edges = self.total_edge_weight - total_internal_edges
        return (cut_edges + non_edges_inside) / total_possible_edges

    def _compute_global_distance(self, idx: int, centroid_type: str) -> float:
        """Computes normalized distance to global reference."""
        centroids = self.get_centroids(idx, centroid_type)
        labels = self.labels_list[idx]
        
        if len(centroids) == 0:
            return float('inf')

        # Select global reference
        global_center = self.global_X if centroid_type in ['X', 'X_w'] else self.global_A

        # Compute weighted average distance
        unique_labels = np.setdiff1d(np.unique(labels), [-1])
        distances = []
        weights = []
        
        for k in unique_labels:
            mask = (labels == k)
            size = np.sum(mask)
            
            # Find centroid index
            idx_k = np.where(unique_labels == k)[0][0]
            dist = np.linalg.norm(centroids[idx_k] - global_center)
            
            distances.append(dist)
            weights.append(size)
            
        norm_factor = self._get_normalization_factors().get(centroid_type, 1.0)
        return np.average(distances, weights=weights) / norm_factor


class MultiDatasetComparator(BaseComparator):
    def __init__(self, datasets: List[Tuple[np.ndarray, Union[csr_matrix, np.ndarray]]], 
                 labels_list: List[np.ndarray], **kwargs):
        """Compares clusterings on different datasets with same structure."""
        super().__init__(labels_list, **kwargs)
        self.datasets = [
            (X, A if issparse(A) else csr_matrix(A)) 
            for X, A in datasets
        ]
        
        # Validate inputs
        if len(datasets) != len(labels_list):
            raise ValueError("Number of datasets must match number of clusterings")
        
        for i, ((X, A), labels) in enumerate(zip(self.datasets, labels_list)):
            n_samples = len(labels)
            if X.shape[0] != n_samples:
                raise ValueError(f"Dataset {i}: Feature matrix has {X.shape[0]} samples, expected {n_samples}")
            if A.shape[0] != n_samples or A.shape[1] != n_samples:
                raise ValueError(f"Dataset {i}: Adjacency matrix shape {A.shape} doesn't match labels {n_samples}")
        
        # Precompute global references, graphs and total weights
        self.global_refs = []
        self.graphs = []
        self.total_weights = []
        
        for X, A in self.datasets:
            self.global_refs.append({
                'X': np.mean(X, axis=0),
                'A': np.array(A.mean(axis=0)).flatten()
            })
            self.graphs.append(nx.from_scipy_sparse_array(A))
            self.total_weights.append(A.sum() / 2)

    def get_centroids(self, i: int, centroid_type: str) -> np.ndarray:
        """Retrieves centroids for clustering i on its dataset."""
        cache_key = (i, centroid_type)
        if cache_key in self.centroids_cache:
            return self.centroids_cache[cache_key]
            
        cache_path = self.cache_dir / "centroids" / f"clustering_{i}_{centroid_type}.npy"
        if cache_path.exists():
            centroids = np.load(cache_path)
        else:
            X, A = self.datasets[i]
            centroids = self._compute_centroids(X, A, self.labels_list[i], centroid_type)
            np.save(cache_path, centroids)
            
        self.centroids_cache[cache_key] = centroids
        return centroids
        
    def _compute_centroids(self, X: np.ndarray, A: csr_matrix, labels: np.ndarray, 
                          centroid_type: str) -> np.ndarray:
        """Computes centroids for a clustering solution on its dataset."""
        unique_labels = np.unique(labels)
        centroids = []

        for k in unique_labels:
            if k == -1:  # Skip noise
                continue

            mask = (labels == k)
            cluster_size = np.sum(mask)

            if cluster_size == 0:  # Skip empty clusters
                continue

            try:
                if centroid_type == 'X':
                    centroids.append(np.mean(X[mask], axis=0))

                elif centroid_type == 'A':
                    cluster_adj = A[mask].mean(axis=0)
                    if issparse(cluster_adj):
                        cluster_adj = cluster_adj.toarray().flatten()
                    centroids.append(cluster_adj)

                elif centroid_type == 'X_w':
                    weights = np.array(A[mask].sum(axis=1)).flatten() + 1e-10
                    centroids.append(np.average(X[mask], axis=0, weights=weights))

                elif centroid_type == 'A_w':
                    weights = np.array(A[mask].sum(axis=1)).flatten() + 1e-10
                    cluster_adj = A[mask].multiply(weights[:, np.newaxis]).mean(axis=0)
                    if issparse(cluster_adj):
                        cluster_adj = cluster_adj.toarray().flatten()
                    centroids.append(cluster_adj)

            except Exception as e:
                warnings.warn(f"Failed to compute {centroid_type} centroid for cluster {k}: {str(e)}")

        return np.array(centroids) if centroids else np.empty((0, X.shape[1]))
    
    def compute_global_metrics(self) -> Dict[str, Any]:
        """Computes global metrics for all clustering solutions."""
        if self.global_metrics is not None:
            return self.global_metrics

        # Compute in parallel
        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as executor:
            futures = {executor.submit(self._compute_metrics_for_clustering, i): i 
                      for i in range(self.n_clusterings)}
            
            results = [None] * self.n_clusterings
            for future in as_completed(futures):
                i = futures[future]
                results[i] = future.result()

        # Unpack results
        silhouette_metrics = [r[0] for r in results]
        global_dist_metrics = [r[1] for r in results]
        graph_metrics = self._compute_graph_metrics()

        self.global_metrics = {
            'silhouette': silhouette_metrics,
            'global_distance': global_dist_metrics,
            'graph_metrics': graph_metrics
        }
        return self.global_metrics

    def _compute_metrics_for_clustering(self, i: int) -> Tuple[Dict, Dict]:
        """Compute metrics for a single clustering solution."""
        X, A = self.datasets[i]
        labels = self.labels_list[i]
        
        # Silhouette scores
        sil_x = silhouette_score(X, labels)
        try:
            D = 1 - normalize(A, norm='l1', axis=1)
            sil_a = silhouette_score(D, labels, metric='precomputed')
        except Exception:
            sil_a = -1
            
        # Global distances
        dist_x = self._compute_global_distance(i, 'X')
        dist_a = self._compute_global_distance(i, 'A')
        dist_xw = self._compute_global_distance(i, 'X_w')
        dist_aw = self._compute_global_distance(i, 'A_w')
        
        return (
            {'features': sil_x, 'adjacency': sil_a},
            {'X': dist_x, 'A': dist_a, 'X_w': dist_xw, 'A_w': dist_aw}
        )

    def _compute_graph_metrics(self) -> Dict[str, List]:
        """Computes graph-specific metrics for all clusterings."""
        if self.graph_metrics is not None:
            return self.graph_metrics

        metrics = {
            'modularity': [],
            'conductance': [],
            'coverage': [],
            'performance': []
        }

        # Compute in parallel
        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as executor:
            futures = {executor.submit(self._compute_graph_metrics_for_dataset, i): i 
                      for i in range(self.n_clusterings)}
            
            results = [None] * self.n_clusterings
            for future in as_completed(futures):
                i = futures[future]
                results[i] = future.result()

        # Unpack results
        for i in range(self.n_clusterings):
            mod, cond, cov, perf = results[i]
            metrics['modularity'].append(mod)
            metrics['conductance'].append(cond)
            metrics['coverage'].append(cov)
            metrics['performance'].append(perf)

        self.graph_metrics = metrics
        return metrics

    def _compute_graph_metrics_for_dataset(self, i: int) -> Tuple[float, float, float, float]:
        """Computes graph metrics for a single clustering solution."""
        _, A = self.datasets[i]
        labels = self.labels_list[i]
        G = self.graphs[i]
        total_weight = self.total_weights[i]
        
        # Modularity
        communities = self._labels_to_communities(labels)
        modularity = nx.algorithms.community.modularity(G, communities) if communities else -1

        # Conductance and Coverage
        conductance, coverage = self._compute_conductance_and_coverage(A, labels, total_weight)

        # Performance (only for binary graphs)
        performance = self._compute_performance(A, labels) if self._is_binary(A) else np.nan

        return modularity, conductance, coverage, performance

    def _is_binary(self, A: csr_matrix) -> bool:
        """Check if graph is binary (weights are 0 or 1)."""
        return np.all(np.isin(A.data, [0, 1]))

    def _labels_to_communities(self, labels: np.ndarray) -> List[List[int]]:
        """Converts labels to communities, excluding noise."""
        communities = {}
        for node_id, label in enumerate(labels):
            if label == -1:
                continue
            communities.setdefault(label, []).append(node_id)
        return list(communities.values())

    def _compute_conductance_and_coverage(self, A: csr_matrix, labels: np.ndarray, 
                                         total_edge_weight: float) -> Tuple[float, float]:
        """Computes average conductance and coverage for clustering."""
        unique_labels = np.setdiff1d(np.unique(labels), [-1])
        conductances = []
        total_internal_edges = 0.0

        for k in unique_labels:
            mask = (labels == k)
            cluster_size = np.sum(mask)

            if cluster_size == 0 or cluster_size == len(labels):
                continue

            # Compute cut and volume
            cut_size = A[mask][:, ~mask].sum()
            volume = A[mask].sum()
            internal_edges = A[mask][:, mask].sum() / 2  # Undirected
            total_internal_edges += internal_edges

            if volume > 0:
                conductances.append(cut_size / volume)

        avg_conductance = np.mean(conductances) if conductances else float('inf')
        coverage = total_internal_edges / total_edge_weight if total_edge_weight > 0 else 0
        return avg_conductance, coverage

    def _compute_performance(self, A: csr_matrix, labels: np.ndarray) -> float:
        """Computes performance metric for binary graphs."""
        unique_labels = np.setdiff1d(np.unique(labels), [-1])
        n_samples = len(labels)
        total_possible_edges = n_samples * (n_samples - 1) / 2
        non_edges_inside = 0
        total_internal_edges = 0

        for k in unique_labels:
            mask = (labels == k)
            cluster_size = np.sum(mask)

            if cluster_size < 2:
                continue

            # Possible edges inside cluster
            possible_edges = cluster_size * (cluster_size - 1) / 2

            # Actual edges inside cluster
            internal_edges = A[mask][:, mask].sum() / 2  # Undirected
            total_internal_edges += internal_edges
            non_edges_inside += (possible_edges - internal_edges)

        cut_edges = (A.sum() / 2) - total_internal_edges
        return (cut_edges + non_edges_inside) / total_possible_edges

    def _compute_global_distance(self, idx: int, centroid_type: str) -> float:
        """Computes normalized distance to global reference."""
        centroids = self.get_centroids(idx, centroid_type)
        labels = self.labels_list[idx]
        global_ref = self.global_refs[idx]['X' if centroid_type in ['X', 'X_w'] else 'A']
        
        if len(centroids) == 0:
            return float('inf')

        # Compute weighted average distance
        unique_labels = np.setdiff1d(np.unique(labels), [-1])
        distances = []
        weights = []
        
        for k in unique_labels:
            mask = (labels == k)
            size = np.sum(mask)
            
            # Find centroid index
            idx_k = np.where(unique_labels == k)[0][0]
            dist = np.linalg.norm(centroids[idx_k] - global_ref)
            
            distances.append(dist)
            weights.append(size)
            
        norm_factor = self._get_normalization_factors().get(centroid_type, 1.0)
        return np.average(distances, weights=weights) / norm_factor