# Файл: metrics/quality.py
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from core.interfaces import Metric
from config.registries import register_metric


@register_metric('silhouette')
class SilhouetteScore(Metric):
    """Вычисляет силуэтный коэффициент для оценки качества кластеризации."""
    
    def calculate(self, data: pd.DataFrame, labels: pd.Series, model_data: dict) -> float:
        """Вычисляет метрику на основе данных и предсказанных кластеров.
        
        Args:
            data (pd.DataFrame): Исходные данные
            labels (pd.Series): Метки кластеров
            model_data (dict): Дополнительные данные модели
            
        Returns:
            float: Значение метрики
        """
        return silhouette_score(data, labels)


@register_metric('inertia')
class InertiaScore(Metric):
    """Вычисляет инерцию (сумму квадратов расстояний) для K-Means."""
    
    def calculate(self, data: pd.DataFrame, labels: pd.Series, model_data: dict) -> float:
        """Вычисляет метрику на основе данных и центроидов кластеров.
        
        Args:
            data (pd.DataFrame): Исходные данные
            labels (pd.Series): Метки кластеров
            model_data (dict): Должен содержать ключ 'centroids'
            
        Returns:
            float: Значение метрики
            
        Raises:
            ValueError: Если центроиды не найдены в model_data
        """
        centroids = model_data.get('centroids')
        if centroids is None:
            raise ValueError("Centroids required for inertia calculation")
        return sum(np.linalg.norm(data.values - centroids[label])**2 for label in labels)