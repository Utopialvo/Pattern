# Файл: models/sklearn_cluster.py
import networkx as nx
import numpy as np
import joblib
from sklearn.cluster import SpectralClustering
from config.registries import register_model
from core.interfaces import ClusterModel
from typing import Dict, Any

class NetworkClusterModel(ClusterModel):
    """Base class for graph-based clustering models"""
    
    def predict(self, data_loader) -> np.ndarray:
        _, similarity = data_loader.full_data()
        return self._predict(similarity.values)

    @property
    def model_data(self) -> dict:
        return {}
    
    @classmethod
    def load(cls, path: str) -> 'NetworkClusterModel':
        """Base load method"""
        pass

    def save(self, path: str) -> None:
        """Base save method"""
        pass

@register_model(
    name='louvain',
    params_help={
        'resolution': 'Community size control (float, default=1.0)',
        'threshold': 'Merge threshold for communities (float, default=0.0000001)',
        'max_level': 'Max recursion level (int, default=15)'
    }
)
class LouvainCluster(NetworkClusterModel):
    """Louvain community detection implementation using NetworkX"""
    
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.labels_ = None

    def fit(self, data_loader) -> None:
        _, adj_matrix = data_loader.full_data()
        G = nx.from_numpy_array(adj_matrix.values)
        
        communities = nx.community.louvain_communities(
            G,
            resolution=self.params.get('resolution', 1.0),
            threshold=self.params.get('threshold', 0.0000001),
            max_level=self.params.get('max_level', 15),
            seed=0
        )
        
        self.labels_ = np.zeros(adj_matrix.shape[0], dtype=int)
        for idx, comm in enumerate(communities):
            self.labels_[list(comm)] = idx

    def _predict(self, adj_matrix: np.ndarray) -> np.ndarray:
        return self.labels_

    def save(self, path: str) -> None:
        data = {
            'params': self.params,
            'labels': self.labels_
        }
        joblib.dump(data, path)

    @classmethod
    def load(cls, path: str) -> 'LouvainCluster':
        data = joblib.load(path)
        model = cls(data['params'])
        model.labels_ = data['labels']
        return model

@register_model(
    name='spectral',
    params_help={
        'n_clusters': 'Number of clusters (positive integer)',
        'n_neighbors': 'Neighbors for affinity matrix (int, optional)',
        'assign_labels': 'Label assignment strategy (kmeans, discretize, etc)',
        'degree': 'Degree of the polynomial kernel (int, optional)'
    }
)
class SpectralGraphCluster(NetworkClusterModel):
    """Spectral clustering implementation for graph data"""
    
    def __init__(self, params: Dict[str, Any]):
        self.model = SpectralClustering(
            n_jobs=-1,
            random_state=0,
            affinity='precomputed',
            **params
        )

    def fit(self, data_loader) -> None:
        _, adj_matrix = data_loader.full_data()
        self.model.fit(adj_matrix.values)

    def _predict(self, adj_matrix: np.ndarray) -> np.ndarray:
        return self.model.labels_
        
    def save(self, path: str) -> None:
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str) -> 'SpectralGraphCluster':
        _model = joblib.load(path)
        model = cls({})
        model.model = _model
        return model