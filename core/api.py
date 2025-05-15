# Файл: core/api.py
from __future__ import annotations
from typing import Union, Dict, Any, List, Optional
from pyspark.sql import SparkSession
from config.registries import MODEL_REGISTRY, METRIC_REGISTRY
from core.interfaces import ComponentFactory, ClusterModel
from preprocessing.normalizers import SparkNormalizer, PandasNormalizer
from core.factory import factory
import pandas as pd

def train_pipeline(
    features_src: Union[str, pd.DataFrame],
    similarity_src: Optional[Union[str, pd.DataFrame]] = None,
    algorithm: str = 'kmeans',
    param_grid: Optional[Dict[str, list[Any]]] = None,
    normalizer: Optional[Union[str, dict, SparkNormalizer, PandasNormalizer]] = None,
    metric: str = 'silhouette',
    optimizer: str = 'grid',
    custom_factory: Optional[ComponentFactory] = None,
    spark: Optional[SparkSession] = None,
    **kwargs
) -> ClusterModel:
    """Universal training pipeline for clustering models.
    
    Combines key stages:
    1. Data loading and preparation
    2. Hyperparameter optimization
    3. Final model training
    
    Args:
        features_src: Primary data source (path or DataFrame)
        similarity_src: Secondary data source for similarity matrices (optional)
        algorithm: Model algorithm identifier (default: 'kmeans')
        param_grid: Hyperparameter search space (parameter -> values)
        normalizer: Normalization configuration (path/object/dict) (optional)
        metric: Optimization metric identifier (default: 'silhouette')
        optimizer: Search strategy ('grid' or 'random') (default: 'grid')
        custom_factory: Alternative component factory (optional)
        spark: Spark session for distributed processing (optional)
        
    Returns:
        Trained model with optimized parameters
        
    Raises:
        ValueError: For invalid parameters or configurations
        TypeError: For incorrect parameter types
        
    Example:
        >>> model = train_pipeline(
        >>>     features_src="data.csv",
        >>>     algorithm="dbscan",
        >>>     param_grid={"eps": [0.3, 0.5]}
        >>> )
    """
    # Configuration setup
    used_factory = custom_factory or factory
    param_grid = param_grid or {}
    data_sources = [features_src] + ([similarity_src] if similarity_src else [])

    # Normalizer initialization
    if isinstance(normalizer, str):
        if spark is not None:
            normalizer = SparkNormalizer.load(normalizer)
        else:
            normalizer = PandasNormalizer.load(normalizer)

    # Parameter validation
    for param, values in param_grid.items():
        if not isinstance(values, list):
            raise TypeError(f"Param {param} must be list, got {type(values)}")
        if not values:
            raise ValueError(f"Empty values for {param}")

    # Component initialization
    model_class = MODEL_REGISTRY[algorithm]['class']
    data_loader = used_factory.create_loader(
        data_src=data_sources,
        spark=spark,
        normalizer=normalizer,
        **kwargs
    )
    
    # Optimization process
    optimizer = used_factory.create_optimizer(optimizer)
    metric = used_factory.create_metric(metric)
    best_params = optimizer.find_best(
        model_class=model_class,
        data_loader=data_loader,
        param_grid=param_grid,
        metric=metric
    )
    
    # Final model training
    best_model = used_factory.create_model(algorithm, best_params)
    data_loader = used_factory.create_loader(
        data_src=data_sources,
        spark=spark,
        normalizer=normalizer,
        **kwargs
    )
    best_model.fit(data_loader=data_loader)
    return best_model