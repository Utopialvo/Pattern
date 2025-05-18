# Файл: models/attribute.py
import joblib
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from numbers import Number
from core.interfaces import ClusterModel, DataLoader
from config.registries import register_model

class SklearnClusterModel(ClusterModel):
    """Base class for scikit-learn clustering implementations."""
    
    def fit(self, data_loader: DataLoader) -> None:
        """Fit model to data from loader."""
        features, _ = data_loader.full_data()
        self.model.fit(features)
        self.labels_ = self.model.predict(features)

    def predict(self, data_loader: DataLoader) -> np.ndarray:
        """Predict cluster labels for new data."""
        features, _ = data_loader.full_data()
        return self.model.predict(features)

    def save(self, path: str) -> None:
        """Persist model to disk."""
        data = {
            'params': self.model.get_params(),
            'labels': self.model.labels_
        }
        joblib.dump(data, path)

    @classmethod
    def load(cls, path: str) -> 'SklearnClusterModel':
        """Load model from disk."""
        data = joblib.load(path)
        model = cls(data['params'])
        model.model = model.model.set_params(**params)
        model.labels_ = data['labels']
        return model

@register_model(
    name='kmeans',
    params_help={
        'n_clusters': 'Number of clusters (positive integer)',
        'init': 'Initialization method [k-means++, random]',
        'max_iter': 'Maximum iterations (positive integer)'
    }
)
class SklearnKMeans(SklearnClusterModel):
    """K-Means implementation using scikit-learn."""
    
    def __init__(self, params: dict):
        self._validate_params(params)
        self.model = KMeans(**params)
        self._centroids = None
        self.labels_ = None

    def _validate_params(self, params: dict) -> None:
        """Validate KMeans-specific parameters."""
        if not isinstance(params.get('n_clusters'), int) or params['n_clusters'] < 1:
            raise ValueError("n_clusters must be positive integer")

    def fit(self, data_loader: DataLoader) -> None:
        super().fit(data_loader)
        self._centroids = self.model.cluster_centers_

    @property
    def model_data(self) -> dict:
        return {'centroids': self._centroids}

@register_model(
    name='dbscan',
    params_help={
        'eps': 'Maximum distance between samples (positive float)',
        'min_samples': 'Minimum samples in neighborhood (positive integer)'
    }
)
class SklearnDBSCAN(SklearnClusterModel):
    """DBSCAN implementation using scikit-learn."""
    
    def __init__(self, params: dict):
        self._validate_params(params)
        self.model = DBSCAN(**params)
        self.labels_ = None
        
    def _validate_params(self, params: dict) -> None:
        """Validate DBSCAN-specific parameters."""
        if not isinstance(params.get('eps'), Number) or params['eps'] <= 0:
            raise ValueError("eps must be positive number")
        if not isinstance(params.get('min_samples'), int) or params['min_samples'] < 1:
            raise ValueError("min_samples must be positive integer")

    @property
    def model_data(self) -> dict:
        return {}