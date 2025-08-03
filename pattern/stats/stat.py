# Файл: stats/stat.py
import os
from pathlib import Path
from typing import Union
import numpy as np
import pandas as pd
from pattern.core.interfaces import DataStatistics, DataLoader
from pattern.stats.statanalyzer import StatisticsAnalyzer
from pyspark.sql import DataFrame as SparkDataFrame


def _is_pandas(obj) -> bool:
    """If object is Pandas DataFrame?"""
    return isinstance(obj, pd.DataFrame)
    
class Statistics(DataStatistics):
    def __init__(self, stat_path=None):
        super().__init__()
        self.stat_path = stat_path
        self.analyzer = StatisticsAnalyzer(save_path=self.stat_path)

    
    def compute_statistics(self,
                  data_loader: DataLoader,
                  labels: Union[pd.Series, pd.DataFrame, SparkDataFrame], 
                  alpha: float = 0.05) -> float:
        
        if _is_pandas(data_loader.features):
            print('Create analyzer')
            print('analyze')
            self.analyzer.analyze(data_loader.features, labels, alpha)
            self.analyzer.print_report()
            self.report = self.analyzer.take_results()