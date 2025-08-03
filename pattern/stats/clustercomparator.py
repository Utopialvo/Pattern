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

class ClusteringComparator:
    def __init__(self, X, A, labels_list, output_dir="reports", alpha=0.5, jaccard_threshold=0.0):
        """
        Compares multiple clustering solutions using both feature and graph-based metrics.
        
        Parameters:
        X (np.ndarray): Feature matrix (n_samples, n_features)
        A (scipy.sparse or np.ndarray): Adjacency matrix (n_samples, n_samples)
        labels_list (list): List of clustering label arrays
        output_dir (str): Base directory for all output reports
        alpha (float): Weight for combining feature and graph distances (0-1)
        jaccard_threshold (float): Similarity threshold for cluster matching
        """
        # Validate input dimensions and properties
        self._validate_inputs(X, A, labels_list)
        
        self.X = X
        self.A = A
        self.labels_list = labels_list
        self.n_clusterings = len(labels_list)
        self.alpha = alpha
        self.jaccard_threshold = jaccard_threshold
        
        # Create unique timestamped output directory
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir) / self.timestamp
        self.cache_dir = self.output_dir / "cache"
        (self.cache_dir / "centroids").mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "mappings").mkdir(parents=True, exist_ok=True)
        
        # Convert to sparse format for efficient graph operations
        if not issparse(self.A):
            self.A = csr_matrix(self.A)
        
        # Compute global reference points
        self.global_X = np.mean(X, axis=0)
        self.global_A = np.array(self.A.mean(axis=0)).flatten()
        
        # Cache graph and total edge weight
        self.G = nx.from_scipy_sparse_array(self.A)
        self.total_edge_weight = self.A.sum() / 2  # For undirected graphs
        
        # Initialize caches
        self.centroids_cache = {}
        self.normalization_factors = None
        self.global_metrics = None
        self.graph_metrics = None
        self.full_report = None
    
    def _validate_inputs(self, X, A, labels_list):
        """Validates input dimensions and properties"""
        n_samples = X.shape[0]
        
        if A.shape[0] != n_samples or A.shape[1] != n_samples:
            raise ValueError(f"Adjacency matrix shape {A.shape} doesn't match feature matrix {X.shape}")
        
        for i, labels in enumerate(labels_list):
            if len(labels) != n_samples:
                raise ValueError(f"Labels[{i}] length {len(labels)} doesn't match samples {n_samples}")
            if np.any(labels < -1):
                raise ValueError(f"Labels[{i}] contain negative values < -1")
            if not isinstance(labels, np.ndarray):
                raise TypeError(f"Labels[{i}] must be numpy array")
        
        # Check graph symmetry
        if (A != A.T).sum() > 1e-10:
            warnings.warn("Adjacency matrix is not symmetric - forcing symmetry")
            self.A = (self.A + self.A.T) / 2

    def get_centroids(self, i, centroid_type):
        """Retrieves centroids from cache or computes them"""
        cache_path = self.cache_dir / "centroids" / f"clustering_{i}_{centroid_type}.npy"
        
        # Memory cache
        if (i, centroid_type) in self.centroids_cache:
            return self.centroids_cache[(i, centroid_type)]
        
        # Disk cache
        if cache_path.exists():
            centroids = np.load(cache_path)
        else:
            centroids = self._compute_centroids(self.labels_list[i], centroid_type)
            np.save(cache_path, centroids)
        
        self.centroids_cache[(i, centroid_type)] = centroids
        return centroids
    
    def _compute_centroids(self, labels, centroid_type):
        """Computes centroids for a clustering solution"""
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
    
    def _get_normalization_factors(self):
        """Computes normalization factors using median of pairwise distances"""
        if self.normalization_factors is not None:
            return self.normalization_factors
        
        cache_path = self.cache_dir / "normalization_factors.pkl"
        if cache_path.exists():
            self.normalization_factors = joblib.load(cache_path)
            return self.normalization_factors
            
        centroid_types = ['X', 'A', 'X_w', 'A_w']
        self.normalization_factors = {}
        
        for ct in centroid_types:
            all_centroids = []
            for i in range(self.n_clusterings):
                centroids = self.get_centroids(i, ct)
                if len(centroids) > 0:
                    all_centroids.append(centroids)
            
            if not all_centroids:
                self.normalization_factors[ct] = 1.0
                continue
                
            try:
                # Ensure consistent dimensionality
                dims = [c.shape[1] for c in all_centroids]
                if len(set(dims)) > 1:
                    warnings.warn(f"Inconsistent centroid dimensions for {ct}: {dims}")
                    self.normalization_factors[ct] = 1.0
                    continue
                    
                stacked = np.vstack(all_centroids)
                
                # Handle single centroid case
                if stacked.shape[0] < 2:
                    self.normalization_factors[ct] = 1.0
                    continue
                    
                # Compute pairwise distances
                distances = cdist(stacked, stacked)
                np.fill_diagonal(distances, np.nan)  # Ignore diagonal
                
                # Use median + small epsilon to avoid division by zero
                median_dist = np.nanmedian(distances)
                self.normalization_factors[ct] = median_dist + 1e-10
                
            except Exception as e:
                warnings.warn(f"Could not compute normalization for {ct}: {str(e)}")
                self.normalization_factors[ct] = 1.0
        
        joblib.dump(self.normalization_factors, cache_path)
        return self.normalization_factors
    
    def compute_global_metrics(self):
        """Computes global metrics for all clustering solutions"""
        if self.global_metrics is not None:
            return self.global_metrics
            
        metrics = {
            'silhouette': [],
            'global_distance': [],
            'graph_metrics': self._compute_graph_metrics()
        }
        
        for i, labels in enumerate(self.labels_list):
            # Silhouette scores
            sil_x = silhouette_score(self.X, labels)
            
            try:
                D = 1 - normalize(self.A, norm='l1', axis=1)
                sil_a = silhouette_score(D, labels, metric='precomputed')
            except Exception as e:
                sil_a = -1
                warnings.warn(f"Graph silhouette failed: {str(e)}")
            
            # Global distances
            dist_x = self._compute_global_distance(i, 'X')
            dist_a = self._compute_global_distance(i, 'A')
            dist_xw = self._compute_global_distance(i, 'X_w')
            dist_aw = self._compute_global_distance(i, 'A_w')
            
            metrics['silhouette'].append({
                'features': sil_x,
                'adjacency': sil_a
            })
            
            metrics['global_distance'].append({
                'X': dist_x,
                'A': dist_a,
                'X_w': dist_xw,
                'A_w': dist_aw
            })
        
        self.global_metrics = metrics
        return metrics
    
    def _compute_graph_metrics(self):
        """Computes graph-specific metrics"""
        if self.graph_metrics is not None:
            return self.graph_metrics
            
        metrics = {
            'modularity': [],
            'conductance': [],
            'coverage': [],
            'performance': []
        }
        
        for labels in self.labels_list:
            # Modularity
            communities = self._labels_to_communities(labels)
            modularity = nx.algorithms.community.modularity(self.G, communities) if communities else -1
            
            # Conductance and Coverage
            conductance, coverage = self._compute_conductance_and_coverage(labels)
            
            # Performance (only for binary graphs)
            performance = self._compute_performance(labels) if self._is_binary() else np.nan
            
            metrics['modularity'].append(modularity)
            metrics['conductance'].append(conductance)
            metrics['coverage'].append(coverage)
            metrics['performance'].append(performance)
        
        self.graph_metrics = metrics
        return metrics
    
    def _is_binary(self):
        """Check if graph is binary (weights are 0 or 1)"""
        if issparse(self.A):
            data = self.A.data
        else:
            data = self.A.ravel()
        return np.all(np.isin(data, [0, 1]))
    
    def _labels_to_communities(self, labels):
        """Converts labels to communities"""
        communities = {}
        for i, label in enumerate(labels):
            if label == -1:
                continue
            communities.setdefault(label, []).append(i)
        return list(communities.values())
    
    def _compute_conductance_and_coverage(self, labels):
        """Computes conductance and coverage"""
        unique_labels = np.unique(labels)
        total_conductance = 0.0
        valid_clusters = 0
        total_internal_edges = 0.0
        
        for k in unique_labels:
            if k == -1:
                continue
                
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
                conductance = cut_size / volume
                total_conductance += conductance
                valid_clusters += 1
        
        avg_conductance = total_conductance / valid_clusters if valid_clusters > 0 else float('inf')
        coverage = total_internal_edges / self.total_edge_weight if self.total_edge_weight > 0 else 0
        
        return avg_conductance, coverage
    
    def _compute_performance(self, labels):
        """Computes performance metric for binary graphs"""
        unique_labels = np.unique(labels)
        total_possible_edges = len(labels) * (len(labels) - 1) / 2
        non_edges_inside = 0
        total_internal_edges = 0
        
        for k in unique_labels:
            if k == -1:
                continue
                
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
        performance = (cut_edges + non_edges_inside) / total_possible_edges
        return performance
    
    def _compute_global_distance(self, idx, centroid_type):
        """Computes normalized distance to global reference"""
        centroids = self.get_centroids(idx, centroid_type)
        if len(centroids) == 0:
            return float('inf')
        
        # Select global reference
        global_center = self.global_X if centroid_type in ['X', 'X_w'] else self.global_A
        
        # Normalize distances
        norm_factors = self._get_normalization_factors()
        distances = [np.linalg.norm(c - global_center) / norm_factors[centroid_type] 
                     for c in centroids]
        return np.mean(distances)
    
    def get_cluster_mapping(self, i, j):
        """
        Finds optimal cluster mapping between two clusterings.
        """
        cache_path = self.cache_dir / "mappings" / f"mapping_{i}_{j}.pkl"
        
        # Load from cache if available
        if cache_path.exists():
            return joblib.load(cache_path)
        
        labels_i = self.labels_list[i]
        labels_j = self.labels_list[j]
        unique_i = np.unique(labels_i)
        unique_j = np.unique(labels_j)
        
        # Filter out noise clusters
        unique_i = unique_i[unique_i != -1]
        unique_j = unique_j[unique_j != -1]
        
        # Compute Jaccard similarity matrix
        similarity_matrix = np.zeros((len(unique_i), (len(unique_j))))
        
        for idx_i, k_i in enumerate(unique_i):
            mask_i = (labels_i == k_i)
            for idx_j, k_j in enumerate(unique_j):
                mask_j = (labels_j == k_j)
                
                intersection = np.sum(mask_i & mask_j)
                union = np.sum(mask_i | mask_j)
                
                similarity_matrix[idx_i, idx_j] = intersection / union if union > 0 else 0
        
        # Find optimal assignment
        row_idx, col_idx = linear_sum_assignment(-similarity_matrix)
        mappings = []
        
        # Filter mappings by similarity threshold
        for r, c in zip(row_idx, col_idx):
            similarity = similarity_matrix[r, c]
            if similarity >= self.jaccard_threshold:
                mappings.append({
                    'cluster_i': int(unique_i[r]),
                    'cluster_j': int(unique_j[c]),
                    'similarity': similarity
                })
        
        # Save to cache
        joblib.dump(mappings, cache_path)
        return mappings
    
    def compute_cluster_distances(self, i, j, cluster_i, cluster_j):
        """
        Computes distances between specific clusters.
        """
        distances = {}
        norm_factors = self._get_normalization_factors()
        
        for c_type in ['X', 'A', 'X_w', 'A_w']:
            centroids_i = self.get_centroids(i, c_type)
            centroids_j = self.get_centroids(j, c_type)
            
            # Skip if no centroids available
            if len(centroids_i) == 0 or len(centroids_j) == 0:
                distances[c_type] = float('inf')
                continue
            
            # Find cluster positions
            try:
                unique_i = np.unique(self.labels_list[i])
                unique_j = np.unique(self.labels_list[j])
                unique_i = unique_i[unique_i != -1]
                unique_j = unique_j[unique_j != -1]
                
                idx_i = np.where(unique_i == cluster_i)[0][0]
                idx_j = np.where(unique_j == cluster_j)[0][0]
                
                # Check dimensionality
                if centroids_i[idx_i].shape != centroids_j[idx_j].shape:
                    warnings.warn(f"Shape mismatch for {c_type} centroids: "
                                 f"{centroids_i[idx_i].shape} vs {centroids_j[idx_j].shape}")
                    distances[c_type] = float('inf')
                    continue
                    
                # Compute and normalize distance
                raw_dist = np.linalg.norm(centroids_i[idx_i] - centroids_j[idx_j])
                norm_dist = raw_dist / norm_factors[c_type]
                distances[c_type] = norm_dist
                
            except Exception as e:
                distances[c_type] = float('inf')
                warnings.warn(f"Distance computation failed for {c_type}: {str(e)}")
        
        # Compute combined distance
        combined = self.alpha * distances.get('X', 1.0) + (1 - self.alpha) * distances.get('A', 1.0)
        
        return {
            'distances': distances,
            'combined_distance': combined
        }
    
    def generate_pairwise_report(self, i, j):
        """
        Generates detailed comparison report for two clusterings.
        """
        # Compute global similarity metrics
        ari = adjusted_rand_score(self.labels_list[i], self.labels_list[j])
        nmi = normalized_mutual_info_score(self.labels_list[i], self.labels_list[j])
        
        # Get cluster mappings
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
                    'distances': dist_data['distances'],
                    'combined_distance': dist_data['combined_distance']
                })
            except Exception as e:
                warnings.warn(f"Distance computation failed for clusters {mapping['cluster_i']} and {mapping['cluster_j']}: {str(e)}")
        
        # Identify unmapped clusters
        unique_i = np.unique(self.labels_list[i])
        unique_j = np.unique(self.labels_list[j])
        unique_i = unique_i[unique_i != -1]
        unique_j = unique_j[unique_j != -1]
        
        mapped_i = {m['cluster_i'] for m in mappings}
        mapped_j = {m['cluster_j'] for m in mappings}
        
        unmapped_i = [int(k) for k in unique_i if k not in mapped_i]
        unmapped_j = [int(k) for k in unique_j if k not in mapped_j]
        
        return {
            'clustering_i': i,
            'clustering_j': j,
            'global_metrics': {
                'ARI': ari,
                'NMI': nmi
            },
            'cluster_mappings': mappings,
            'cluster_distances': cluster_distances,
            'unmapped_clusters': {
                f'clustering_{i}': unmapped_i,
                f'clustering_{j}': unmapped_j
            }
        }
    
    def generate_full_report(self):
        """
        Generates comprehensive comparison report for all clustering pairs.
        """
        start_time = time.time()
        print(f"Starting full report generation at {self.timestamp}")
        
        # Compute global metrics
        global_metrics = self.compute_global_metrics()
        
        report = {
            'timestamp': self.timestamp,
            'parameters': {
                'alpha': self.alpha,
                'jaccard_threshold': self.jaccard_threshold
            },
            'global_comparison': {
                'ari_matrix': np.zeros((self.n_clusterings, self.n_clusterings)),
                'nmi_matrix': np.zeros((self.n_clusterings, self.n_clusterings)),
                'silhouette_scores': global_metrics['silhouette'],
                'global_distances': global_metrics['global_distance'],
                'graph_metrics': global_metrics['graph_metrics']
            },
            'pairwise_reports': {}
        }
        
        # Compute similarity matrices in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all pairwise tasks
            future_to_index = {}
            for i in range(self.n_clusterings):
                for j in range(self.n_clusterings):
                    if i != j:
                        future = executor.submit(
                            adjusted_rand_score, 
                            self.labels_list[i], 
                            self.labels_list[j]
                        )
                        future_to_index[future] = ('ari', i, j)
                        
                        future = executor.submit(
                            normalized_mutual_info_score, 
                            self.labels_list[i], 
                            self.labels_list[j]
                        )
                        future_to_index[future] = ('nmi', i, j)
        
            # Process completed tasks
            for future in as_completed(future_to_index):
                metric_type, i, j = future_to_index[future]
                result = future.result()
                if metric_type == 'ari':
                    report['global_comparison']['ari_matrix'][i, j] = result
                else:
                    report['global_comparison']['nmi_matrix'][i, j] = result
        
        # Set diagonal to perfect similarity
        np.fill_diagonal(report['global_comparison']['ari_matrix'], 1.0)
        np.fill_diagonal(report['global_comparison']['nmi_matrix'], 1.0)
        
        # Generate pairwise reports in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}
            for i in range(self.n_clusterings):
                for j in range(i + 1, self.n_clusterings):
                    future = executor.submit(self.generate_pairwise_report, i, j)
                    futures[future] = (i, j)
            
            # Collect results
            for future in as_completed(futures):
                i, j = futures[future]
                report_key = f"clustering_{i}_vs_{j}"
                report['pairwise_reports'][report_key] = future.result()
        
        # Save and store report
        self.full_report = report
        elapsed = time.time() - start_time
        print(f"Report generated in {elapsed:.2f} seconds. Output directory: {self.output_dir}")
        return report
    
    def export_to_dataframes(self):
        """
        Exports full report to a collection of pandas DataFrames.
        
        Returns:
        dict: Dictionary of DataFrames with keys:
            - global_metrics
            - similarity_matrices
            - pairwise_mappings
            - cluster_distances
            - unmapped_clusters
        """
        if self.full_report is None:
            self.generate_full_report()
        
        dfs = {}
        
        # Global metrics per clustering
        global_data = []
        for i, (sil, gdist, mod, cond, cov, perf) in enumerate(zip(
            self.full_report['global_comparison']['silhouette_scores'],
            self.full_report['global_comparison']['global_distances'],
            self.full_report['global_comparison']['graph_metrics']['modularity'],
            self.full_report['global_comparison']['graph_metrics']['conductance'],
            self.full_report['global_comparison']['graph_metrics']['coverage'],
            self.full_report['global_comparison']['graph_metrics']['performance']
        )):
            global_data.append({
                'clustering_index': i,
                'silhouette_features': sil['features'],
                'silhouette_adjacency': sil['adjacency'],
                'global_dist_X': gdist['X'],
                'global_dist_A': gdist['A'],
                'global_dist_X_w': gdist['X_w'],
                'global_dist_A_w': gdist['A_w'],
                'modularity': mod,
                'conductance': cond,
                'coverage': cov,
                'performance': perf
            })
        dfs['global_metrics'] = pd.DataFrame(global_data)
        
        # Similarity matrices
        dfs['similarity_matrices'] = {
            'ARI': pd.DataFrame(
                self.full_report['global_comparison']['ari_matrix'],
                index=range(self.n_clusterings),
                columns=range(self.n_clusterings)
            ),
            'NMI': pd.DataFrame(
                self.full_report['global_comparison']['nmi_matrix'],
                index=range(self.n_clusterings),
                columns=range(self.n_clusterings)
            )
        }
        
        # Pairwise mappings and distances
        mappings_data = []
        distances_data = []
        unmapped_data = []
        
        for key, report in self.full_report['pairwise_reports'].items():
            i, j = report['clustering_i'], report['clustering_j']
            
            # Cluster mappings
            for mapping in report['cluster_mappings']:
                mappings_data.append({
                    'clustering_pair': key,
                    'clustering_i': i,
                    'clustering_j': j,
                    'cluster_i': mapping['cluster_i'],
                    'cluster_j': mapping['cluster_j'],
                    'similarity': mapping['similarity']
                })
            
            # Cluster distances
            for dist in report['cluster_distances']:
                distances_data.append({
                    'clustering_pair': key,
                    'clustering_i': i,
                    'clustering_j': j,
                    'cluster_i': dist['cluster_i'],
                    'cluster_j': dist['cluster_j'],
                    'dist_X': dist['distances'].get('X', np.nan),
                    'dist_A': dist['distances'].get('A', np.nan),
                    'dist_X_w': dist['distances'].get('X_w', np.nan),
                    'dist_A_w': dist['distances'].get('A_w', np.nan),
                    'combined_distance': dist['combined_distance']
                })
            
            # Unmapped clusters
            for cluster in report['unmapped_clusters'][f'clustering_{i}']:
                unmapped_data.append({
                    'clustering_pair': key,
                    'clustering_index': i,
                    'cluster_id': cluster
                })
            for cluster in report['unmapped_clusters'][f'clustering_{j}']:
                unmapped_data.append({
                    'clustering_pair': key,
                    'clustering_index': j,
                    'cluster_id': cluster
                })
        
        dfs['pairwise_mappings'] = pd.DataFrame(mappings_data)
        dfs['cluster_distances'] = pd.DataFrame(distances_data)
        dfs['unmapped_clusters'] = pd.DataFrame(unmapped_data)
        
        # Save all DataFrames to CSV
        (self.output_dir / "dataframes").mkdir(exist_ok=True)
        for name, df in dfs.items():
            if isinstance(df, dict):
                for matrix_name, matrix_df in df.items():
                    path = self.output_dir / "dataframes" / f"{name}_{matrix_name}.csv"
                    matrix_df.to_csv(path, index=True)
            else:
                path = self.output_dir / "dataframes" / f"{name}.csv"
                df.to_csv(path, index=False)
        
        print(f"DataFrames saved to: {self.output_dir / 'dataframes'}")
        return dfs


# import numpy as np
# from sklearn.datasets import make_blobs
# from sklearn.cluster import KMeans, AgglomerativeClustering

# kmeans3 = KMeans(n_clusters=16, n_init=1, random_state=42).fit_predict(X)
# kmeans4 = KMeans(n_clusters=15, n_init=1, random_state=42).fit_predict(X)
# kmeans4_1 = KMeans(n_clusters=15, init='random', n_init=1, random_state=23).fit_predict(X)
# agglo5 = AgglomerativeClustering(n_clusters=15).fit_predict(X)
# agglo4 = AgglomerativeClustering(n_clusters=17).fit_predict(X)

# labels_list = [kmeans3, kmeans4, kmeans4_1, agglo5, agglo4]

# comparator = ClusteringComparator(
#     X=X,
#     A=A,
#     labels_list=labels_list,
#     output_dir="clustering_reports",
#     alpha=0.5,
#     jaccard_threshold=0.0
# )

# full_report = comparator.generate_full_report()
# dfs = comparator.export_to_dataframes()

# print("Global Metrics:")
# print(dfs['global_metrics'])

# print("\nARI Matrix:")
# print(dfs['similarity_matrices']['ARI'])

# print("\nCluster Mappings:")
# print(dfs['pairwise_mappings'].head())