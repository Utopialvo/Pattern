# Файл: core/abstractions.py
import pandas as pd
from typing import Any, Dict, Union
from core.interfaces import ComponentFactory, ClusterModel, Metric, DataLoader, Optimizer
from config.registries import MODEL_REGISTRY, METRIC_REGISTRY
from data.loaders import PandasDataLoader, SparkDataLoader
from optimization.strategies import GridSearch, RandomSearch
from preprocessing.normalizers import Normalizer

from models import *
from metrics import *
from pyspark.sql import SparkSession


class DefaultFactory(ComponentFactory):
    """Фабрика по умолчанию, использующая зарегистрированные компоненты."""
    
    def create_model(self, 
                     identifier: str, 
                     config: Dict[str, Any]) -> ClusterModel:
        """Создает экземпляр модели с валидацией параметров.
        
        Args:
            identifier (str): Имя алгоритма из MODEL_REGISTRY
            config (dict): Конкретные параметры для инициализации модели (не сетка!)
        
        Returns:
            ClusterModel: Готовый к использованию экземпляр модели
        """
        if identifier not in MODEL_REGISTRY:
            available = list(MODEL_REGISTRY.keys())
            raise ValueError(f"Unknown algorithm: {identifier}. Available: {available}")
        # Валидация обязательных параметров
        required_params = set(MODEL_REGISTRY[identifier]['params_help'].keys())
        missing = required_params - config.keys()
        if missing:
            raise ValueError(f"Missing required parameters for {identifier}: {missing}")
        # Создание экземпляра модели
        return MODEL_REGISTRY[identifier]['class'](config)
    
    def create_metric(self, identifier: str) -> Metric:
        """Фабрика метрик через регистр."""
        if identifier not in METRIC_REGISTRY:
            raise ValueError(f"Unknown metric: {identifier}. Available: {list(METRIC_REGISTRY.keys())}")
        return METRIC_REGISTRY[identifier]()
    
    def create_optimizer(self, identifier: str, **kwargs) -> Optimizer:
        """Автоматическое создание Optimizer."""
        optimizers = {
            'grid': GridSearch,
            'random': RandomSearch
        }
        return optimizers[identifier](**kwargs)
    
    def create_loader(self, 
                      data_src: Union[pd.DataFrame, str], 
                      normalizer: Union[Normalizer, str] = None,
                      spark: SparkSession = None, 
                      **kwargs) -> DataLoader:
        """Автоматическое создание DataLoader по типу данных.
            Args:
            data_src: Источник данных
            normalizer: Путь к файлу нормализатора или объект Normalizer
            spark: SparkSession
            **kwargs: Доп. параметры загрузчика
        Returns:
            DataLoader с интегрированной нормализацией
        """
        base_loader = super().create_loader(data_src, **kwargs)
        
        if normalizer:
            if isinstance(normalizer, str):
                normalizer = Normalizer
                normalizer.load(normalizer)
            if spark:
                return NormalizingDataLoader(base_loader = base_loader, normalizer = normalizer, spark=spark, **kwargs)
            else:
                return NormalizingDataLoader(base_loader = base_loader, normalizer = normalizer, **kwargs)
        else:
            if spark:
                return SparkDataLoader(data_src, spark, **kwargs)
            elif isinstance(data_src, pd.DataFrame) or isinstance(data_src, str):
                return PandasDataLoader(data_src, **kwargs)
            raise ValueError("Unsupported data source type")

factory = DefaultFactory()

