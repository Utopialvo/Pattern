# Файл: metrics/quality.py
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from core.interfaces import Metric
from config.registries import register_metric


@register_metric('silhouette')
class SilhouetteScore(Metric):
    """Calculates silhouette coefficient for clustering quality assessment."""
    
    def calculate(self, data_loader, labels: pd.Series, model_data: dict) -> float:
        """Compute metric using features and cluster assignments.
        
        Args:
            data_loader: Source containing features DataFrame
            labels: Cluster assignments for each sample
            model_data: Additional model information (unused)
            
        Returns:
            Silhouette score between -1 and 1
        """
        return silhouette_score(data_loader.features, labels)

@register_metric('inertia')
class InertiaScore(Metric):
    """Calculates sum of squared distances to nearest cluster center."""
    
    def calculate(self, data_loader, labels: pd.Series, model_data: dict) -> float:
        """Compute total intra-cluster variance.
        
        Args:
            data_loader: Source containing features DataFrame
            labels: Cluster assignments for each sample
            model_data: Must contain 'centroids' key with cluster centers
            
        Returns:
            Total within-cluster sum of squares
            
        Raises:
            KeyError: If centroids missing from model_data
        """
        features = data_loader.features.values
        centroids = model_data.get('centroids')
        if centroids is None:
            raise ValueError("Centroids required for inertia calculation")
        # Vectorized distance calculation
        distances = np.linalg.norm(features - centroids[labels], axis=1)
        return np.sum(distances ** 2)