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
import networkx as nx
import pandas as pd
import time
from datetime import datetime
from abc import ABC, abstractmethod
from itertools import combinations
from functools import lru_cache
from typing import List, Dict, Tuple, Union, Any
from tqdm import tqdm

class BaseComparator(ABC):
    def __init__(self, labels_list: List[np.ndarray], output_dir: str = "reports", jaccard_threshold: float = 0.0):
        self.labels_list = labels_list
        self.n_clusterings = len(labels_list)
        self.jaccard_threshold = jaccard_threshold
        self._validate_labels()
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir) / self.timestamp
        self.cache_dir = self.output_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.centroids_cache = {}
        self.normalization_factors = None
        self.global_metrics = None
        self.full_report = None

    def _validate_labels(self) -> None:
        n_samples = len(self.labels_list[0])
        for i, labels in enumerate(self.labels_list):
            if len(labels) != n_samples:
                raise ValueError(f"Labels[{i}] length {len(labels)} doesn't match first clustering {n_samples}")
            if np.any(labels < -1):
                raise ValueError(f"Labels[{i}] contain negative values < -1")
            if not isinstance(labels, np.ndarray):
                raise TypeError(f"Labels[{i}] must be numpy array")

    @abstractmethod
    def get_data_for_clustering(self, i: int) -> Tuple[np.ndarray, csr_matrix]:
        pass

    def get_centroids(self, i: int, centroid_type: str) -> np.ndarray:
        cache_key = (i, centroid_type)
        if cache_key in self.centroids_cache:
            return self.centroids_cache[cache_key]

        cache_path = self.cache_dir / f"centroids_{i}_{centroid_type}.npy"
        if cache_path.exists():
            centroids = np.load(cache_path)
        else:
            X, A = self.get_data_for_clustering(i)
            centroids = self._compute_centroids(X, A, self.labels_list[i], centroid_type)
            np.save(cache_path, centroids)

        self.centroids_cache[cache_key] = centroids
        return centroids

    def _compute_centroids(self, X: np.ndarray, A: csr_matrix, labels: np.ndarray, 
                          centroid_type: str) -> np.ndarray:
        unique_labels = np.unique(labels)
        centroids = []

        for k in unique_labels:
            if k == -1:
                continue

            mask = (labels == k)
            if not np.any(mask):
                continue

            try:
                if centroid_type == 'X':
                    centroids.append(np.mean(X[mask], axis=0))
                elif centroid_type == 'A':
                    centroids.append(A[mask].mean(axis=0).A1)
                elif centroid_type == 'X_w':
                    weights = A[mask].sum(axis=1).A1 + 1e-10
                    centroids.append(np.average(X[mask], axis=0, weights=weights))
                elif centroid_type == 'A_w':
                    weights = A[mask].sum(axis=1).A1 + 1e-10
                    centroids.append(A[mask].multiply(weights[:, None]).mean(axis=0).A1)
            except Exception:
                continue

        return np.array(centroids) if centroids else np.empty((0, X.shape[1]))
    
    def _get_normalization_factors(self) -> Dict[str, float]:
        if self.normalization_factors is not None:
            return self.normalization_factors

        centroid_types = ['X', 'A', 'X_w', 'A_w']
        self.normalization_factors = {}
        
        for c_type in centroid_types:
            all_centroids = []
            cluster_sizes = []
            
            for i in range(self.n_clusterings):
                centroids = self.get_centroids(i, c_type)
                if len(centroids) == 0:
                    continue
                    
                all_centroids.append(centroids)
                labels = self.labels_list[i]
                sizes = [np.sum(labels == k) for k in np.unique(labels) if k != -1]
                cluster_sizes.append(sizes)
                
            if not all_centroids:
                self.normalization_factors[c_type] = 1.0
                continue
                
            weighted_dists = []
            n = len(all_centroids)
            
            for i in range(n):
                for j in range(i + 1, n):
                    dists = cdist(all_centroids[i], all_centroids[j])
                    weights = np.outer(cluster_sizes[i], cluster_sizes[j])
                    weighted_avg = np.average(dists, weights=weights)
                    weighted_dists.append(weighted_avg)
            
            self.normalization_factors[c_type] = np.median(weighted_dists) + 1e-10 if weighted_dists else 1.0
        
        return self.normalization_factors

    def get_cluster_mapping(self, i: int, j: int) -> List[Dict]:
        cache_path = self.cache_dir / f"mapping_{i}_{j}.pkl"
        if cache_path.exists():
            return joblib.load(cache_path)

        labels_i = self.labels_list[i]
        labels_j = self.labels_list[j]
        unique_i = np.setdiff1d(np.unique(labels_i), [-1])
        unique_j = np.setdiff1d(np.unique(labels_j), [-1])

        similarity_matrix = np.zeros((len(unique_i), len(unique_j)))
        for idx_i, k_i in enumerate(unique_i):
            mask_i = labels_i == k_i
            for idx_j, k_j in enumerate(unique_j):
                mask_j = labels_j == k_j
                intersection = np.sum(mask_i & mask_j)
                union = np.sum(mask_i | mask_j)
                similarity_matrix[idx_i, idx_j] = intersection / union if union > 0 else 0

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

        joblib.dump(mappings, cache_path)
        return mappings

    def compute_cluster_distances(self, i: int, j: int, cluster_i: int, cluster_j: int) -> Dict[str, float]:
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
                
                idx_i = np.where(unique_i == cluster_i)[0][0]
                idx_j = np.where(unique_j == cluster_j)[0][0]
                
                raw_dist = np.linalg.norm(centroids_i[idx_i] - centroids_j[idx_j])
                distances[c_type] = raw_dist / norm_factors[c_type]
            except Exception:
                distances[c_type] = float('inf')

        return {
            'feature_distance': distances.get('X', float('inf')),
            'topology_distance': distances.get('A', float('inf')),
            'feature_weighted_distance': distances.get('X_w', float('inf')),
            'topology_weighted_distance': distances.get('A_w', float('inf'))
        }

    def generate_pairwise_report(self, i: int, j: int) -> Dict[str, Any]:
        ari = adjusted_rand_score(self.labels_list[i], self.labels_list[j])
        nmi = normalized_mutual_info_score(self.labels_list[i], self.labels_list[j])
        mappings = self.get_cluster_mapping(i, j)

        cluster_distances = []
        for mapping in mappings:
            dist_data = self.compute_cluster_distances(
                i, j, mapping['cluster_i'], mapping['cluster_j']
            )
            cluster_distances.append({
                'cluster_i': mapping['cluster_i'],
                'cluster_j': mapping['cluster_j'],
                'similarity': mapping['similarity'],
                'distances': dist_data
            })

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

    def compute_global_metrics(self) -> Dict[str, Any]:
        if self.global_metrics is not None:
            return self.global_metrics

        results = []
        for i in range(self.n_clusterings):
            results.append(self._compute_metrics_for_clustering(i))
        
        self.global_metrics = {
            'silhouette': [r[0] for r in results],
            'global_distance': [r[1] for r in results],
            'graph_metrics': self._compute_graph_metrics()
        }
        return self.global_metrics

    def _compute_metrics_for_clustering(self, i: int) -> Tuple[Dict, Dict]:
        X, A = self.get_data_for_clustering(i)
        labels = self.labels_list[i]
        
        sil_x = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else -1
        try:
            D = 1 - normalize(A, norm='l1', axis=1)
            sil_a = silhouette_score(D, labels, metric='precomputed') if len(np.unique(labels)) > 1 else -1
        except Exception:
            sil_a = -1
            
        dist_x = self._compute_global_distance(i, 'X')
        dist_a = self._compute_global_distance(i, 'A')
        dist_xw = self._compute_global_distance(i, 'X_w')
        dist_aw = self._compute_global_distance(i, 'A_w')
        
        return (
            {'features': sil_x, 'adjacency': sil_a},
            {'X': dist_x, 'A': dist_a, 'X_w': dist_xw, 'A_w': dist_aw}
        )

    def _compute_graph_metrics(self) -> Dict[str, List]:
        metrics = {
            'modularity': [],
            'conductance': [],
            'coverage': []
        }

        for i in range(self.n_clusterings):
            X, A = self.get_data_for_clustering(i)
            labels = self.labels_list[i]
            G = nx.from_scipy_sparse_array(A)
            total_weight = A.sum() / 2
            
            mod, cond, cov = self._compute_graph_metrics_single(A, labels, G, total_weight)
            metrics['modularity'].append(mod)
            metrics['conductance'].append(cond)
            metrics['coverage'].append(cov)

        return metrics

    def _compute_graph_metrics_single(self, A: csr_matrix, labels: np.ndarray, 
                                    G: nx.Graph, total_weight: float) -> Tuple[float, float, float]:
        unique_labels = np.setdiff1d(np.unique(labels), [-1])
        if not unique_labels.size:
            return -1, float('inf'), 0

        communities = []
        conductances = []
        total_internal_edges = 0.0
        total_volume = 2 * total_weight  # Total graph volume

        for k in unique_labels:
            mask = (labels == k)
            cluster_nodes = np.where(mask)[0]
            if len(cluster_nodes) == 0 or len(cluster_nodes) == len(labels):
                continue

            communities.append(cluster_nodes.tolist())
            
            # Compute cluster metrics
            volume_S = A[mask].sum()
            internal_edges = A[mask][:, mask].sum() / 2
            total_internal_edges += internal_edges
            cut_size = A[mask][:, ~mask].sum()
            
            volume_notS = total_volume - volume_S
            min_volume = min(volume_S, volume_notS)
            
            if min_volume > 0:
                conductances.append(cut_size / min_volume)
            else:
                conductances.append(float('inf'))

        # Compute metrics
        modularity = nx.algorithms.community.modularity(G, communities) if communities else -1
        avg_conductance = np.mean(conductances) if conductances else float('inf')
        coverage = total_internal_edges / total_weight if total_weight > 0 else 0
        
        return modularity, avg_conductance, coverage

    def _compute_global_distance(self, idx: int, centroid_type: str) -> float:
        centroids = self.get_centroids(idx, centroid_type)
        labels = self.labels_list[idx]
        
        if len(centroids) == 0:
            return float('inf')

        # Get global reference
        X, A = self.get_data_for_clustering(idx)
        if centroid_type in ['X', 'X_w']:
            global_ref = np.mean(X, axis=0)
        else:
            global_ref = A.mean(axis=0).A1

        # Compute weighted distances
        unique_labels = np.setdiff1d(np.unique(labels), [-1])
        distances = []
        weights = []
        
        for k in unique_labels:
            mask = (labels == k)
            size = np.sum(mask)
            if size == 0:
                continue
                
            try:
                idx_k = np.where(unique_labels == k)[0][0]
                dist = np.linalg.norm(centroids[idx_k] - global_ref)
                distances.append(dist)
                weights.append(size)
            except Exception:
                continue
                
        if not distances:
            return float('inf')
            
        norm_factor = self._get_normalization_factors().get(centroid_type, 1.0)
        return np.average(distances, weights=weights) / norm_factor

    def generate_full_report(self) -> Dict[str, Any]:
        if self.full_report is not None:
            return self.full_report

        start_time = time.time()
        print(f"Starting full report generation at {self.timestamp}")

        global_metrics = self.compute_global_metrics()
        n = self.n_clusterings
        
        ari_matrix = np.eye(n)
        nmi_matrix = np.eye(n)
        pairwise_reports = {}
        
        pairs = list(combinations(range(n), 2))
        for i, j in tqdm(pairs, desc="Computing pairwise metrics"):
            ari = adjusted_rand_score(self.labels_list[i], self.labels_list[j])
            nmi = normalized_mutual_info_score(self.labels_list[i], self.labels_list[j])
            ari_matrix[i, j] = ari_matrix[j, i] = ari
            nmi_matrix[i, j] = nmi_matrix[j, i] = nmi
            pairwise_reports[f"clustering_{i}_vs_{j}"] = self.generate_pairwise_report(i, j)

        self.full_report = {
            'timestamp': self.timestamp,
            'parameters': {'jaccard_threshold': self.jaccard_threshold},
            'global_comparison': {
                'ari_matrix': ari_matrix,
                'nmi_matrix': nmi_matrix,
                'global_metrics': global_metrics
            },
            'pairwise_reports': pairwise_reports
        }
        
        elapsed = time.time() - start_time
        print(f"Report generated in {elapsed:.2f} seconds. Output directory: {self.output_dir}")
        return self.full_report

    def export_to_dataframes(self) -> Dict[str, pd.DataFrame]:
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
                'coverage': graph_metrics['coverage'][i]
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

        for key, pairwise_report in tqdm(report['pairwise_reports'].items(), desc="Exporting reports"):
            if "error" in pairwise_report:
                continue
                
            i = pairwise_report['clustering_i']
            j = pairwise_report['clustering_j']

            for mapping in pairwise_report['cluster_mappings']:
                mappings_data.append({
                    'clustering_pair': key,
                    'clustering_i': i,
                    'clustering_j': j,
                    'cluster_i': mapping['cluster_i'],
                    'cluster_j': mapping['cluster_j'],
                    'similarity': mapping['similarity']
                })

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
        for name, df in tqdm(dfs.items(), desc="Saving dataframes"):
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
        super().__init__(labels_list, **kwargs)
        self.X = X
        self.A = A if issparse(A) else csr_matrix(A)
        self._validate_data()

    def _validate_data(self) -> None:
        n_samples = len(self.labels_list[0])
        if self.X.shape[0] != n_samples:
            raise ValueError(f"Feature matrix shape {self.X.shape} doesn't match labels {n_samples}")
        if self.A.shape[0] != n_samples or self.A.shape[1] != n_samples:
            raise ValueError(f"Adjacency matrix shape {self.A.shape} doesn't match labels {n_samples}")
            
    def get_data_for_clustering(self, i: int) -> Tuple[np.ndarray, csr_matrix]:
        return self.X, self.A


class MultiDatasetComparator(BaseComparator):
    def __init__(self, datasets: List[Tuple[np.ndarray, Union[csr_matrix, np.ndarray]]], 
                 labels_list: List[np.ndarray], **kwargs):
        super().__init__(labels_list, **kwargs)
        self.datasets = [
            (X, A if issparse(A) else csr_matrix(A)) 
            for X, A in datasets
        ]
        self._validate_data()

    def _validate_data(self) -> None:
        if len(self.datasets) != len(self.labels_list):
            raise ValueError("Number of datasets must match number of clusterings")
        
        for i, ((X, A), labels) in enumerate(zip(self.datasets, self.labels_list)):
            n_samples = len(labels)
            if X.shape[0] != n_samples:
                raise ValueError(f"Dataset {i}: Feature matrix has {X.shape[0]} samples, expected {n_samples}")
            if A.shape[0] != n_samples or A.shape[1] != n_samples:
                raise ValueError(f"Dataset {i}: Adjacency matrix shape {A.shape} doesn't match labels {n_samples}")
        
    def get_data_for_clustering(self, i: int) -> Tuple[np.ndarray, csr_matrix]:
        return self.datasets[i]