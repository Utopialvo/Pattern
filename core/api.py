# Файл: core/api.py
from typing import Union, Dict, Any, List
import pandas as pd
from pyspark.sql import SparkSession
from config.registries import MODEL_REGISTRY, METRIC_REGISTRY
from core.interfaces import ComponentFactory, ClusterModel
from preprocessing.normalizers import Normalizer
from core.factory import factory
import datetime


def train_pipeline(
    data_src: Any,
    algorithm: str,
    param_grid: Dict[str, list],
    metric: str = 'silhouette',
    optimizer: str = 'grid',
    custom_factory: ComponentFactory = None,
    normalizer: str = None,
    spark: SparkSession = None,
    **kwargs) -> ClusterModel:
    """Универсальный пайплайн обучения модели.
    
    Объединяет все этапы:
    1. Создание загрузчика данных
    2. Оптимизация гиперпараметров
    3. Обучение финальной модели
    
    Args:
        data_src (Any): Источник данных (DataFrame или путь)
        algorithm (str): Идентификатор алгоритма
        param_grid (dict): Сетка параметров для оптимизации
        metric (str): Идентификатор метрики (default: 'silhouette')
        optimizer (str): Тип оптимизатора ('grid' или 'random') (default: 'grid')
        custom_factory (ComponentFactory): Кастомная фабрика (optional)
        normalizer (str): Конфигурация нормализации json path (optional)
        spark: SparkSession (optional)
        
    Returns:
        ClusterModel: Обученная модель с лучшими параметрами
        
    Пример:
        >>> model = train_pipeline(df, 'kmeans', {'n_clusters': [3,5]})
    """
    used_factory = custom_factory or factory
    
    
    # Валидация структуры param_grid
    for param, values in param_grid.items():
        if not isinstance(values, list):
            raise TypeError(f"Parameter {param} must be a list of values")
        if len(values) == 0:
            raise ValueError(f"Empty values list for parameter {param}")
    
    # Получение класса модели
    model_class = MODEL_REGISTRY[algorithm]['class']
    data_loader = used_factory.create_loader(data_src, spark = spark, normalizer = normalizer)
    optimizer = used_factory.create_optimizer(optimizer)
    metric = used_factory.create_metric(metric)
    
    # Запуск оптимизации
    best_params = optimizer.find_best(
        model_class=model_class,
        data_loader=data_loader,
        param_grid=param_grid,
        metric=metric
    )
    
    # Создание финальной модели с лучшими параметрами
    best_model = used_factory.create_model(algorithm, best_params)
    data_loader = used_factory.create_loader(data_src, spark = spark, normalizer = normalizer)
    best_model.fit(data_loader = data_loader)
    return best_model