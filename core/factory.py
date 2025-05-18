# Файл: core/factory.py
from typing import Any, Union, Optional, List, Dict
from pyspark.sql import SparkSession
from core.interfaces import ComponentFactory, ClusterModel, Metric, DataLoader, Optimizer, Normalizer, Sampler, DataVis, DataStatistics
from config.registries import MODEL_REGISTRY, METRIC_REGISTRY
from data.loaders import PandasDataLoader, SparkDataLoader
from optimization.strategies import GridSearch, RandomSearch, TPESearch
from preprocessing.normalizers import SparkNormalizer, PandasNormalizer
from preprocessing.samplers import SparkSampler, PandasSampler
from visualization.vis import Visualizer
from stats.stat import Statistics

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

        model = meta['class'](config)
        if not isinstance(model, ClusterModel):
            raise TypeError(f"{identifier} is not a ClusterModel subclass")
        return model
    
    def create_metric(self, identifier: str) -> Metric:
        """Create metric instance from registry."""
        if (metric_cls := METRIC_REGISTRY.get(identifier)) is None:
            raise ValueError(f"Unknown metric '{identifier}'. Available: {list(METRIC_REGISTRY)}")
        return metric_cls()
    
    def create_optimizer(self, identifier: str, **kwargs) -> Optimizer:
        """Create hyperparameter optimization strategy."""
        strategies = {
            'grid': GridSearch, 
            'random': RandomSearch,
            'tpe': TPESearch
        }
        return strategies[identifier](**kwargs)
    
    def create_sampler(self, data_src: Union[str, List[str]], spark: SparkSession = None) -> Sampler:
        """Create sampler."""
        if isinstance(spark, SparkSession):
            return SparkSampler(data_src = data_src, spark = spark)
        else:
            return PandasSampler(data_src = data_src)

    def create_normalizer(self, 
                          method: Optional[str] = None, 
                          columns: Optional[List[str]] = None, 
                          methods: Optional[Dict[str, str]] = None, 
                          spark: SparkSession = None) -> Normalizer:
        """Create normalizer."""
        if isinstance(spark, SparkSession):
            return SparkNormalizer(method = method, columns = columns, methods = methods, spark = spark)
        else:
            return PandasNormalizer(method = method, columns = columns, methods = methods)

    def create_visualizer(self, plots_path: str) -> DataVis:
        """Create visualizer."""
        if isinstance(plots_path, str):
            visualizer = Visualizer(plots_path)
            return visualizer

    def create_analyser(self, stat_path: str) -> DataStatistics:
        """Create analyser."""
        if isinstance(stat_path, str):
            analyser = Statistics(stat_path)
            return analyser

    def create_loader(self,
                     data_src: Union[str, list, tuple],
                     normalizer: Optional[Union[SparkNormalizer, PandasNormalizer, str]] = None,
                     sampler: Optional[Union[SparkSampler, PandasSampler, str]] = None,
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

