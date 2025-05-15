# Файл: metrics/clustering_metrics.py
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from typing import Dict, Optional

class FuturesClusteringMetrics:
    """A class to compute feature-based clustering metrics.
    
    Metrics include:
    - WB: Within-to-Between cluster variance ratio (K*WSS/BSS)
    - SW: Silhouette Score (cluster cohesion/separation measure)
    - CH: Calinski-Harabasz Score (variance ratio criterion)
    """
    
    def _compute_bss(self, X: np.ndarray, clustering: np.ndarray, centroids: np.ndarray) -> float:
        """Compute Between-Cluster Sum of Squares."""
        global_center = X.mean(axis=0)
        cluster_sizes = np.bincount(clustering, minlength=len(centroids))
        return np.sum(cluster_sizes * np.sum((centroids - global_center) ** 2, axis=1))
    
    def get_metrics(self, X: np.ndarray, clustering: np.ndarray, centroids: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Compute clustering metrics for given data and cluster assignments.
        
        Args:
            X: Feature matrix of shape (N, D)
            clustering: Cluster labels array of shape (N,)
            centroids: Optional precomputed centroids of shape (K, D)
            
        Returns:
            Dictionary with 'WB', 'SW', and 'CH' metrics
        """
        
        n_clusters = len(np.unique(clustering))
        
        if centroids is None:
            centroids = np.array([X[clustering == i].mean(axis=0) for i in range(n_clusters)])
        
        wss = np.sum((X - centroids[clustering]) ** 2)
        bss = self._compute_bss(X, clustering, centroids)
        wb = (n_clusters * wss / bss) if bss != 0 else np.inf
        
        try:
            sw = silhouette_score(X, clustering)
        except ValueError:
            sw = np.nan
            
        ch = calinski_harabasz_score(X, clustering)
        
        return {'WB': wb, 'SW': sw, 'CH': ch}
        
class FuturesClusteringMetricsSpark:
    """to do"""
    def get_metrics(self, features, clustering, centroids=None) -> Dict[str, float]:
        raise NotImplementedError("Spark version of FuturesClusteringMetrics is not implemented yet")


class AdjacencyClusteringMetrics:
    """A class to compute graph-based clustering metrics using adjacency matrices."""
    
    def get_metric(self, adjacency_matrix, clustering):
        """Calculate graph-theoretic clustering metrics.
        
        Args:
            adjacency_matrix (np.ndarray): Square adjacency matrix of the graph.
            clustering (np.ndarray): Cluster labels array of shape (N,).
            
        Returns:
            dict: Dictionary containing:
                - 'ANUI' (float): Adjusted Normalized Uniformity Isolability
                - 'AVU' (float): Average Volume Uniformity
                - 'AVI' (float): Average Volume Isolability
                - 'modularity' (float): Standard graph modularity
                - 'density_modularity' (float): Density modularity
        """
        clusters = np.unique(clustering)
        k = len(clusters)
        if k == 0:
            return {k: 0.0 for k in ['ANUI', 'AVU', 'AVI', 'modularity', 'density_modularity']}
        
        mask_list = [clustering == c for c in clusters]
        S = np.zeros((k, k))
        
        for i in range(k):
            mask_i = mask_list[i]
            for j in range(k):
                mask_j = mask_list[j]
                S[i, j] = adjacency_matrix[np.ix_(mask_i, mask_j)].sum()
        
        sum_edges = adjacency_matrix.sum() / 2
        degrees = adjacency_matrix.sum(axis=1)
        sum_degrees = [degrees[mask].sum() for mask in mask_list]
        community_sizes = [mask.sum() for mask in mask_list]
        
        # Calculate Average Volume Uniformity (AVU)
        avu = 0.0
        for i in range(k):
            sum_i_out = S[i].sum() - S[i, i]
            sum_u_i = 0.0
            for j in range(k):
                if j == i:
                    continue
                sum_j_in = S[:, j].sum() - S[j, j]
                numerator = S[i, j]
                denominator = sum_i_out + sum_j_in - numerator
                contrib = numerator / denominator if denominator != 0 else 0.0
                sum_u_i += contrib
            avu += sum_u_i / k
        
        # Calculate Average Volume Isolability (AVI)
        avi = 0.0
        for i in range(k):
            total = S[i].sum()
            isolability = S[i, i] / total if total != 0 else 0.0
            avi += isolability / k
        
        # Calculate ANUI metric
        anui_denominator = avu + (1.0 / avi) if avi != 0 else 0.0
        anui = 1.0 / anui_denominator if anui_denominator != 0 else 0.0
        
        # Calculate standard modularity
        modularity = 0.0
        if sum_edges != 0:
            for i in range(k):
                sum_internal = S[i, i] / 2  # Internal edges (undirected)
                sum_d = sum_degrees[i]
                modularity += (sum_internal / sum_edges) - (sum_d / (2 * sum_edges)) ** 2
        
        # Calculate density modularity
        density_modularity = 0.0
        if sum_edges != 0:
            for i in range(k):
                sum_internal = S[i, i] / 2
                sum_d = sum_degrees[i]
                size = community_sizes[i]
                term = (sum_internal - (sum_d ** 2) / (4 * sum_edges)) / size if size != 0 else 0.0
                density_modularity += term
        
        return {
            'ANUI': anui,
            'AVU': avu,
            'AVI': avi,
            'modularity': modularity,
            'density_modularity': density_modularity
        }


class AdjacencyClusteringMetricsSpark:
    """to do"""
    def get_metric(self, adjacency_matrix, clustering) -> Dict[str, float]:
        raise NotImplementedError("Spark version of AdjacencyClusteringMetrics is not implemented yet")