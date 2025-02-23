# Файл: config/registries.py
"""Реестры зарегистрированных компонентов системы и функции для их регистрации."""

from typing import Type, Dict, Any
from core.interfaces import ClusterModel, Metric

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {}
METRIC_REGISTRY: Dict[str, Type[Metric]] = {}

def register_model(name: str, params_help: Dict[str, str]):
    """Декоратор для регистрации моделей кластеризации в системе.
    
    Args:
        name (str): Уникальное имя модели для использования в конфигурациях
        params_help (Dict[str, str]): Описание параметров модели для справочной системы
    
    Returns:
        decorator: Декоратор класса модели
    """
    def decorator(cls: Type[ClusterModel]):
        """Внутренний декоратор, который регистрирует класс модели."""
        if not issubclass(cls, ClusterModel):
            raise TypeError(f"{cls.__name__} must be a subclass of ClusterModel")
            
        MODEL_REGISTRY[name] = {
            'class': cls,
            'params_help': params_help
        }
        return cls
    return decorator

def register_metric(name: str):
    """Декоратор для регистрации метрик качества кластеризации.
    
    Args:
        name (str): Уникальное имя метрики для использования в конфигурациях
    
    Returns:
        decorator: Декоратор класса метрики
    """
    def decorator(cls: Type[Metric]):
        """Внутренний декоратор, который регистрирует класс метрики."""
        if not issubclass(cls, Metric):
            raise TypeError(f"{cls.__name__} must be a subclass of Metric")
            
        METRIC_REGISTRY[name] = cls
        return cls
    return decorator