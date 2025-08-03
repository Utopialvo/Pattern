# Файл: core/factory.py
import pandas as pd
from typing import Any, Union, Optional, List, Dict
from pyspark.sql import SparkSession, DataFrame as SparkDataFrame
from pattern.core.interfaces import ComponentFactory, ClusterModel, Metric, DataLoader, Optimizer, Normalizer, Sampler, DataVis, DataStatistics
from pattern.config.registries import get_model_class, get_metric_class
from pattern.data.loaders import PandasDataLoader, SparkDataLoader
from pattern.optimization.strategies import GridSearch, RandomSearch, TPESearch
from pattern.preprocessing.normalizers import SparkNormalizer, PandasNormalizer
from pattern.preprocessing.samplers import SparkSampler, PandasSampler
from pattern.visualization.vis import Visualizer
from pattern.stats.stat import Statistics


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
        cls = get_model_class(identifier)
        return cls(config)
    
    def create_metric(self, identifier: str) -> Metric:
        """Create metric instance from registry."""
        return get_metric_class(identifier)()
    
    def create_optimizer(self, identifier: str, **kwargs) -> Optimizer:
        """Create hyperparameter optimization strategy."""
        strategies = {
            'grid': GridSearch, 
            'random': RandomSearch,
            'tpe': TPESearch
        }
        return strategies[identifier](**kwargs)
    
    def create_sampler(self,
                    features: Optional[Union[str, pd.DataFrame, SparkDataFrame]] = None,
                    similarity: Optional[Union[str, pd.DataFrame, SparkDataFrame]] = None,  
                    spark: SparkSession = None) -> Sampler:
        """Create sampler."""
        if isinstance(spark, SparkSession):
            return SparkSampler(features = features, 
                                similarity=similarity, 
                                spark = spark)
        else:
            return PandasSampler(features = features, 
                                similarity=similarity)

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
                     features: Optional[Union[str, pd.DataFrame, SparkDataFrame]] = None, 
                     similarity: Optional[Union[str, pd.DataFrame, SparkDataFrame]] = None,
                     normalizer: Optional[Union[SparkNormalizer, PandasNormalizer, str]] = None,
                     sampler: Optional[Union[SparkSampler, PandasSampler, str]] = None,
                     spark: Optional[SparkSession] = None,
                     **kwargs) -> DataLoader:
        """Create appropriate data loader with normalization.
        
        Args:
            features: Input data features source(s)
            similarity: Input data similarity source(s)
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
        return loader_cls(
            features=features,
            similarity=similarity,
            normalizer=normalizer,
            sampler=sampler,
            spark=spark,
            **kwargs
        )

factory = DefaultFactory()

