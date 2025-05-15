# Файл: config/registries.py
from typing import Type, Dict, Any
from core.interfaces import ClusterModel, Metric

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {}
METRIC_REGISTRY: Dict[str, Type[Metric]] = {}

def register_model(name: str, params_help: Dict[str, str]):
    """Decorator for registering clustering models in the system.
    
    Args:
        name: Unique model identifier for configurations
        params_help: Parameter descriptions for help system
            (keys = parameter names, values = descriptions)
    
    Returns:
        Class decorator that registers the model
    """
    def decorator(cls: Type[ClusterModel]) -> Type[ClusterModel]:
        """Inner decorator that performs actual class registration."""
        if not issubclass(cls, ClusterModel):
            raise TypeError(f"{cls.__name__} must subclass ClusterModel")
        
        MODEL_REGISTRY[name] = {'class': cls, 'params_help': params_help}
        return cls
    return decorator

def register_metric(name: str):
    """Decorator for registering clustering quality metrics.
    
    Args:
        name: Unique metric identifier for configurations
    
    Returns:
        Class decorator that registers the metric
    """
    def decorator(cls: Type[Metric]) -> Type[Metric]:
        """Inner decorator that performs actual class registration."""
        if not issubclass(cls, Metric):
            raise TypeError(f"{cls.__name__} must subclass Metric")
        
        METRIC_REGISTRY[name] = cls
        return cls
    return decorator