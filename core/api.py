# Файл: core/api.py
from typing import Union, Dict, Any, List
import pandas as pd
from pyspark.sql import SparkSession
from data.loaders import PandasDataLoader, SparkDataLoader
from config.registries import MODEL_REGISTRY, METRIC_REGISTRY
from models import *
from metrics import *

def create_loader(
    data: Union[pd.DataFrame, str],
    spark_session: SparkSession = None,
    batch_size: int = 1000
) -> Union[PandasDataLoader, SparkDataLoader]:
    """Автоматическое создание DataLoader по типу данных."""
    if isinstance(data, pd.DataFrame):
        return PandasDataLoader(data, batch_size)
    
    if spark_session and isinstance(data, str):
        return SparkDataLoader(spark_session, data)
    
    raise ValueError("Unsupported data type or missing Spark session")

def create_model(algorithm: str, params: Dict[str, Any]) -> 'ClusterModel':
    """Создает экземпляр модели с валидацией параметров.
    
    Args:
        algorithm (str): Имя алгоритма из MODEL_REGISTRY
        params (dict): Конкретные параметры для инициализации модели (не сетка!)
    
    Returns:
        ClusterModel: Готовый к использованию экземпляр модели
    """
    if algorithm not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown algorithm: {algorithm}. Available: {available}")
    
    # Валидация обязательных параметров
    required_params = set(MODEL_REGISTRY[algorithm]['params_help'].keys())
    missing = required_params - params.keys()
    if missing:
        raise ValueError(f"Missing required parameters for {algorithm}: {missing}")
    
    # Создание экземпляра модели
    return MODEL_REGISTRY[algorithm]['class'](params)

def create_metric(metric_name: str) -> 'Metric':
    """Фабрика метрик через регистр."""
    if metric_name not in METRIC_REGISTRY:
        raise ValueError(f"Unknown metric: {metric_name}. Available: {list(METRIC_REGISTRY.keys())}")
    return METRIC_REGISTRY[metric_name]()

def train_pipeline(
    data_loader: Union[PandasDataLoader, SparkDataLoader],
    algorithm: str,
    param_grid: Dict[str, List[Any]],
    metric: str = 'silhouette',
    optimizer: str = 'grid'
) -> 'ClusterModel':
    """Универсальный пайплайн обучения с поддержкой оптимизации гиперпараметров.
    
    Args:
        param_grid (dict): Сетка параметров для оптимизации. Пример:
            {
                'n_clusters': [3, 5], 
                'init': ['k-means++', 'random']
            }
    """
    # Валидация структуры param_grid
    for param, values in param_grid.items():
        if not isinstance(values, list):
            raise TypeError(f"Parameter {param} must be a list of values")
        if len(values) == 0:
            raise ValueError(f"Empty values list for parameter {param}")
    
    # Выбор оптимизатора
    if optimizer == 'grid':
        from optimization.strategies import GridSearch
        optimizer = GridSearch()
    elif optimizer == 'random':
        from optimization.strategies import RandomSearch
        optimizer = RandomSearch()
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")
    
    # Получение класса модели
    model_class = MODEL_REGISTRY[algorithm]['class']
    
    # Запуск оптимизации
    best_params = optimizer.find_best(
        model_class=model_class,
        data_loader=data_loader,
        param_grid=param_grid,
        metric=METRIC_REGISTRY[metric]()
    )
    
    # Создание финальной модели с лучшими параметрами
    best_model = create_model(algorithm, best_params)
    best_model.fit(data_loader)
    return best_model