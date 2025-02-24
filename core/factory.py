# Файл: core/abstractions.py
import pandas as pd
from typing import Any, Dict, Union
from core.interfaces import ComponentFactory, ClusterModel, Metric, DataLoader, Optimizer
from config.registries import MODEL_REGISTRY, METRIC_REGISTRY
from data.loaders import PandasDataLoader, SparkDataLoader
from optimization.strategies import GridSearch, RandomSearch

from models import *
from metrics import *
from pyspark.sql import SparkSession


class DefaultFactory(ComponentFactory):
    """Фабрика по умолчанию, использующая зарегистрированные компоненты."""
    
    def create_model(self, identifier: str, config: Dict[str, Any]) -> ClusterModel:
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
    
    def create_loader(self, data_src: Union[pd.DataFrame, str], spark: SparkSession = None, **kwargs) -> DataLoader:
        """Автоматическое создание DataLoader по типу данных."""
        if spark:
            return SparkDataLoader(data_src, spark, **kwargs)
        elif isinstance(data_src, pd.DataFrame) or isinstance(data_src, str):
            return PandasDataLoader(data_src, **kwargs)            
        raise ValueError("Unsupported data source type")


factory = DefaultFactory()


# class CustomFactory(DefaultFactory):
#     def create_model(self, identifier, config):
#         if identifier == "my_model":
#             return MyCustomModel(config)
#         return super().create_model(identifier, config)

# custom_factory = CustomFactory()

