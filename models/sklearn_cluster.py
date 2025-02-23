# Файл: models/sklearn_cluster.py
import pandas as pd
import pickle
from pathlib import Path
from sklearn.cluster import KMeans, DBSCAN
from numbers import Number
from core.interfaces import ClusterModel, DataLoader
from config.registries import register_model


@register_model(
    name='kmeans',
    params_help={
        'n_clusters': 'Number of clusters (positive integer)',
        'init': 'Initialization method [k-means++, random]',
        'max_iter': 'Maximum iterations (positive integer)'
    }
)
class SklearnKMeans(ClusterModel):
    """Реализация K-Means алгоритма с использованием scikit-learn.
    
    Attributes:
        model (KMeans): Объект обученной модели
        _centroids (np.ndarray): Центроиды кластеров
    """
    
    def __init__(self, params: dict):
        """Инициализация модели с валидацией параметров.
        
        Args:
            params (dict): Параметры для инициализации модели
        """
        self._validate_params(params)
        self.model = KMeans(**params)
        self._centroids = None

    def _validate_params(self, params):
        required = {'n_clusters'}
        if not required.issubset(params.keys()):
            raise ValueError(f"Missing parameters: {required}")
        
        if not isinstance(params['n_clusters'], int) or params['n_clusters'] < 1:
            raise ValueError("n_clusters must be positive integer")

    def fit(self, data_loader: DataLoader):
        full_data = pd.concat(data_loader.iter_batches(), ignore_index=True)
        self.model.fit(full_data)
        self._centroids = self.model.cluster_centers_

    def predict(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(self.model.predict(data))

    @property
    def model_data(self) -> dict:
        return {'centroids': self._centroids}

    @classmethod
    def load(cls, path: str) -> 'SklearnKMeans':
        with open(path, 'rb') as f:
            model = pickle.load(f)
        if not isinstance(model, cls):
            raise ValueError("Loaded object is not a SklearnKMeans instance")
        return model

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)


@register_model(
    name='dbscan',
    params_help={
        'eps': 'Maximum distance between samples (positive float)',
        'min_samples': 'Minimum samples in neighborhood (positive integer)'
    }
)
class SklearnDBSCAN(ClusterModel):
    """Реализация DBSCAN алгоритма с использованием scikit-learn.
    
    Attributes:
        model (DBSCAN): Объект обученной модели
    """
    
    def __init__(self, params: dict):
        """Инициализация модели с валидацией параметров.
        
        Args:
            params (dict): Параметры для инициализации модели
        """
        self._validate_params(params)
        self.model = DBSCAN(**params)

    def _validate_params(self, params):
        required = {'eps', 'min_samples'}
        if not required.issubset(params.keys()):
            raise ValueError(f"Missing parameters: {required}")
        
        if not isinstance(params['eps'], Number) or params['eps'] <= 0:
            raise ValueError("eps must be positive number")
        
        if not isinstance(params['min_samples'], int) or params['min_samples'] < 1:
            raise ValueError("min_samples must be positive integer")

    def fit(self, data_loader: DataLoader):
        full_data = pd.concat(data_loader.iter_batches(), ignore_index=True)
        self.model.fit(full_data)

    def predict(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(self.model.fit_predict(data))

    @property
    def model_data(self) -> dict:
        return {}

    @classmethod
    def load(cls, path: str) -> 'SklearnDBSCAN':
        with open(path, 'rb') as f:
            model = pickle.load(f)
        if not isinstance(model, cls):
            raise ValueError("Loaded object is not a SklearnDBSCAN instance")
        return model

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)