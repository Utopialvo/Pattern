# Файл: core/abstractions.py
from typing import Any, Union, Optional
from pyspark.sql import SparkSession
from core.interfaces import ComponentFactory, ClusterModel, Metric, DataLoader, Optimizer
from config.registries import MODEL_REGISTRY, METRIC_REGISTRY
from data.loaders import PandasDataLoader, SparkDataLoader
from optimization.strategies import GridSearch, RandomSearch
from preprocessing.normalizers import SparkNormalizer, PandasNormalizer

from models import *
from metrics import *


class DefaultFactory(ComponentFactory):
    """Default component factory using registered implementations."""
    
    def create_model(self, identifier: str, config: dict[str, Any]) -> ClusterModel:
        """Instantiate a model with parameter validation.
        
        Args:
            identifier: Algorithm name from MODEL_REGISTRY
            config: Concrete initialization parameters (not a grid)
            
        Returns:
            Configured model instance
            
        Raises:
            ValueError: For unknown algorithms or missing parameters
        """
        if (meta := MODEL_REGISTRY.get(identifier)) is None:
            raise ValueError(f"Unknown algorithm '{identifier}'. Available: {list(MODEL_REGISTRY)}")
            
        if missing := set(meta['params_help']) - config.keys():
            raise ValueError(f"Missing required parameters for {identifier}: {missing}")
            
        return meta['class'](config)
    
    def create_metric(self, identifier: str) -> Metric:
        """Create metric instance from registry."""
        if (metric_cls := METRIC_REGISTRY.get(identifier)) is None:
            raise ValueError(f"Unknown metric '{identifier}'. Available: {list(METRIC_REGISTRY)}")
        return metric_cls()
    
    def create_optimizer(self, identifier: str, **kwargs) -> Optimizer:
        """Create hyperparameter optimization strategy."""
        strategies = {'grid': GridSearch, 'random': RandomSearch}
        return strategies[identifier](**kwargs)
    
    def create_loader(self,
                     data_src: Union[str, list, tuple],
                     normalizer: Optional[Union[SparkNormalizer, PandasNormalizer, str]] = None,
                     sampler: Optional['Sampler'] = None,
                     spark: Optional[SparkSession] = None,
                     **kwargs) -> DataLoader:
        """Create appropriate data loader with normalization.
        
        Args:
            data_src: Data source(s) - path(s) or DataFrame(s)
            normalizer: Normalizer instance/config path (optional)
            sampler: Data sampling component (optional)
            spark: Spark session for distributed processing (optional)
            
        Returns:
            Configured data loader with optional normalization
        """
        # Normalizer initialization
        if isinstance(normalizer, str):
            if spark is not None:
                normalizer = SparkNormalizer.load(normalizer)
            else:
                normalizer = PandasNormalizer.load(normalizer)
            
        loader_cls = SparkDataLoader if spark else PandasDataLoader
        data_src = [data_src] if isinstance(data_src, str) else data_src
            
        return loader_cls(
            data_src=data_src,
            normalizer=normalizer,
            sampler=sampler,
            spark=spark,
            **kwargs
        )

factory = DefaultFactory()

