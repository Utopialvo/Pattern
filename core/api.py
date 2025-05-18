# Файл: core/api.py
from __future__ import annotations
from typing import Union, Dict, Any, List, Optional
from pyspark.sql import SparkSession
from config.registries import MODEL_REGISTRY, METRIC_REGISTRY
from core.interfaces import ComponentFactory, ClusterModel
from preprocessing.normalizers import SparkNormalizer, PandasNormalizer
from preprocessing.samplers import SparkSampler, PandasSampler
from core.factory import factory
import pandas as pd


def train_pipeline(
    features_src: Union[str, pd.DataFrame],
    similarity_src: Optional[Union[str, pd.DataFrame]] = None,
    algorithm: str = 'kmeans',
    param_grid: Optional[Dict[str, list[Any]]] = None,
    normalizer: Optional[Union[dict, str]] = None,
    sampler: Optional[Union[dict, str]] = None,
    metric: str = 'silhouette',
    optimizer: str = 'grid',
    plots_path: str = None,
    stat_path: str = None,
    custom_factory: Optional[ComponentFactory] = None,
    spark: Optional[SparkSession] = None,
    **kwargs
) -> ClusterModel:
    """Universal training pipeline for clustering models.
    
    Args:
        features_src: Primary data source (path or DataFrame)
        similarity_src: Secondary data source for similarity matrices (optional)
        algorithm: Model algorithm identifier (default: 'kmeans')
        param_grid: Hyperparameter search space (parameter -> values)
        normalizer: Normalization configuration (path/object/dict) (optional)
        sampler: Sampler configuration (path/object/dict) (optional)
        metric: Optimization metric identifier (default: 'silhouette')
        optimizer: Search strategy ('grid' or 'random') (default: 'grid')
        plots_path: Path to save plosts (path/str) (optional)
        stat_path: Path to save report (path/str) (optional)
        custom_factory: Alternative component factory (optional)
        spark: Spark session for distributed processing (optional)
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

    # # Sampler initialization (placeholder)
    # if isinstance(sampler, str):
    #     if spark is not None:
    #         normalizer = SparkSampler.load(sampler)
    #     else:
    #         normalizer = PandasSampler.load(sampler)
    if isinstance(normalizer, dict):
        normalizer = used_factory.create_normalizer(spark = spark, **normalizer)
    if isinstance(sampler, dict):
        sampler= used_factory.create_sampler(spark = spark, **sampler)

    # Parameter validation
    for param, values in param_grid.items():
        if not isinstance(values, list):
            raise TypeError(f"Param {param} must be list, got {type(values)}")
        if not values:
            raise ValueError(f"Empty values for {param}")

    # Component initialization
    model_class = MODEL_REGISTRY[algorithm]['class']
    data_loader = used_factory.create_loader(
        data_src = data_sources,
        normalizer = normalizer,
        sampler = sampler,
        spark = spark,
        **kwargs
    )
    
    # Optimization process
    optimizer = used_factory.create_optimizer(optimizer)
    metric = used_factory.create_metric(metric)
    
    print('Start find best params...')
    best_params = optimizer.find_best(
        model_class=model_class,
        data_loader=data_loader,
        param_grid=param_grid,
        metric=metric
    )
    print(f"Optimal parameters: {best_params}")
    
    # Final model training
    best_model = used_factory.create_model(algorithm, best_params)
    best_model.fit(data_loader=data_loader)
    print(f"Final train best params")

    if isinstance(plots_path, str):
        print(f"Visualizing")
        visualizer = factory.create_visualizer(plots_path)
        visualizer.visualisation(data_loader, best_model.labels_)

    if isinstance(stat_path, str):
        print(f"Analizing")
        analyser = factory.create_analyser(stat_path)
        analyser.compute_statistics(data_loader, best_model.labels_)
    
    return best_model