# Файл: visualization/vis.py
import os
from pathlib import Path
from typing import Union
import numpy as np
import pandas as pd
from core.interfaces import DataVis, DataLoader
from visualization.type_figs import CombinedVisualizer
from visualization.mirkin_analysis import MirkinAnalysis
from pyspark.sql import DataFrame as SparkDataFrame


def _is_pandas(obj) -> bool:
    """If object is Pandas DataFrame?"""
    return isinstance(obj, pd.DataFrame)
    
class Visualizer(DataVis):
    def __init__(self, save_path=None):
        super().__init__()
        self.save_path = save_path
        self.visualizer = CombinedVisualizer(save_path=self.save_path)
        
    def visualisation(self,
                  data_loader: DataLoader,
                  labels: Union[pd.Series, pd.DataFrame, SparkDataFrame], 
                  model_data: dict = None) -> float:
        
        if _is_pandas(data_loader.features):
            self.visualizer.pairplot(data_loader.features, labels)
            print('Create pairplot')
            self.visualizer.radar_chart(data_loader.features, labels)
            print('Create radar_chart')
            self.visualizer.violin_plots(data_loader.features, labels)
            print('Create violin_plots')
            analyzer = MirkinAnalysis(data_loader.features, labels, save_path=self.save_path)
            analyzer.plot_feature_importance(top_features=5)
            analyzer.plot_feature_deviation()
            print('Create Mirkin Analysis plots')
            
        if _is_pandas(data_loader.similarity_matrix):
            self.visualizer.force_directed(data_loader.similarity_matrix.values, labels)
            print('Create force directed plot')
            self.visualizer.adjacency_heatmap(data_loader.similarity_matrix.values, labels)
            print('Create adjacency heatmap')
            self.visualizer.edge_bundling(data_loader.similarity_matrix.values, labels)
            print('Create edge bundling plot')
            self.visualizer.aggregated_graph(data_loader.similarity_matrix.values, labels)
            print('Create aggregated graph plot')