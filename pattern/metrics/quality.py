# Файл: metrics/quality.py
import numpy as np
import pandas as pd
from pyspark.sql import DataFrame as SparkDF
from sklearn.metrics import silhouette_score
from pattern.core.interfaces import Metric
from pattern.config.registries import register_metric
from pattern.metrics.clustering_metrics import FeaturesClusteringMetrics, FeaturesClusteringMetricsSpark, AdjacencyClusteringMetrics, AdjacencyClusteringMetricsSpark

def _is_spark(obj) -> bool:
    """If object is Spark DataFrame?"""
    return isinstance(obj, SparkDF)
    
def _is_pandas(obj) -> bool:
    """If object is Pandas DataFrame?"""
    return isinstance(obj, pd.DataFrame)

@register_metric('attribute')
class AttributeMetric(Metric):
    """Feature clustering quality metrics"""
    
    def calculate(self, data_loader, labels, model_data) -> float:
        if isinstance(data_loader.features, type(None)):
            return np.nan
            
        if _is_spark(data_loader.features):
            calculator = FeaturesClusteringMetricsSpark()
            features = data_loader.features
        else:
            calculator = FeaturesClusteringMetrics()
            features = data_loader.features.values
        
        centroids = model_data.get('centroids', None)
        
        try:
            return calculator.get_metrics(features, labels, centroids)['SW']
        except NotImplementedError as e:
            print(f"Metric calculation failed: {str(e)}")
            return np.nan

@register_metric('graph')
class GraphMetric(Metric):
    """Graph clustering quality metrics"""
    
    def calculate(self, data_loader, labels, model_data) -> float:
        if isinstance(data_loader.similarity_matrix, type(None)):
            return np.nan
            
        if _is_spark(data_loader.similarity_matrix):
            calculator = AdjacencyClusteringMetricsSpark()
            adj_matrix = data_loader.similarity_matrix
        else:
            calculator = AdjacencyClusteringMetrics()
            adj_matrix = data_loader.similarity_matrix.values
            
        try:
            return calculator.get_metric(adj_matrix, labels)['ANUI']
        except NotImplementedError as e:
            print(f"Metric calculation failed: {str(e)}")
            return np.nan

@register_metric('attribute-graph')
class AttributeGraphMetric(Metric):
    """Combined feature and graph metrics"""
    
    def calculate(self, data_loader, labels, model_data) -> float:
        if not ((_is_spark(data_loader.features) or _is_pandas(data_loader.features))
            or 
            (_is_spark(data_loader.similarity_matrix) or _is_pandas(data_loader.similarity_matrix))):
            return np.nan
        attribute_score = AttributeMetric().calculate(data_loader, labels, model_data)
        graph_score = GraphMetric().calculate(data_loader, labels, model_data)
        return np.nanmean([attribute_score, graph_score])