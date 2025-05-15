# Файл: preprocessing/normalizers.py
from pyspark.sql import functions as F, DataFrame as SparkDF
from typing import Union, Dict, List, Optional
import joblib
import pandas as pd

class BaseNormalizer:
    """Base class for data normalization across frameworks."""
    
    VALID_METHODS = ['minmax', 'zscore', 'range']
    
    def __init__(
        self,
        method: Optional[str] = None,
        columns: Optional[List[str]] = None,
        methods: Optional[Dict[str, str]] = None
    ):
        """Initialize normalization configuration.
        
        Args:
            method: Default normalization method
            columns: Columns to apply default method
            methods: Dictionary of {column: method}
            
        Raises:
            ValueError: For invalid configuration or methods
        """
        self.methods: Dict[str, str] = {}
        self.stats: Dict[str, Dict] = {}
        self.columns: List[str] = []

        if methods is not None:
            if not isinstance(methods, dict):
                raise ValueError("Methods must be a dictionary")
            self._validate_methods(methods.values())
            self.methods = methods
            self.columns = list(methods.keys())
        elif method and columns:
            self._validate_methods([method])
            self.methods = {col: method for col in columns}
            self.columns = columns
        else:
            self.methods = {}
            self.columns = []

    def _validate_methods(self, methods):
        """Validate normalization methods."""
        for method in methods:
            if method not in self.VALID_METHODS:
                raise ValueError(f"Invalid method: {method}. Valid options: {self.VALID_METHODS}")

    def save(self, path: str) -> None:
        """Persist normalization parameters to disk."""
        joblib.dump({
            'methods': self.methods,
            'stats': self.stats,
            'columns': self.columns,
        }, path)

    def load(self, path: str) -> None:
        """Load normalization parameters from disk."""
        data = joblib.load(path)
        self.methods = data['methods']
        self.stats = data['stats']
        self.columns = data['columns']

class SparkNormalizer(BaseNormalizer):
    """PySpark DataFrame normalization implementation."""
    
    def fit(self, df: SparkDF) -> 'SparkNormalizer':
        """Compute normalization statistics from training data."""
        missing = [col for col in self.columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # Build aggregation expressions
        aggs = []
        for col, method in self.methods.items():
            if method == 'minmax':
                aggs.extend([F.min(col), F.max(col)])
            elif method == 'zscore':
                aggs.extend([F.avg(col), F.stddev(col)])
            elif method == 'range':
                aggs.extend([F.min(col), F.max(col), F.avg(col)])

        # Compute statistics
        stats = df.select(*aggs).first()
        
        # Store statistics
        stat_idx = 0
        for col, method in self.methods.items():
            if method == 'minmax':
                self.stats[col] = {'method': method, 'params': stats[stat_idx:stat_idx+2]}
                stat_idx += 2
            elif method == 'zscore':
                self.stats[col] = {'method': method, 'params': stats[stat_idx:stat_idx+2]}
                stat_idx += 2
            elif method == 'range':
                self.stats[col] = {'method': method, 'params': stats[stat_idx:stat_idx+3]}
                stat_idx += 3
        
        return self

    def transform(self, df: SparkDF) -> SparkDF:
        """Apply normalization to DataFrame."""
        if not self.stats:
            raise RuntimeError("Fit the normalizer first")

        for col in self.columns:
            stat = self.stats[col]
            method, params = stat['method'], stat['params']
            
            if method == 'minmax':
                min_val, max_val = params
                df = df.withColumn(col, (F.col(col) - min_val) / (max_val - min_val) if max_val != min_val else F.lit(0.0))
            elif method == 'zscore':
                mean, std = params[0], params[1] or 1e-19
                df = df.withColumn(col, (F.col(col) - mean) / std)
            elif method == 'range':
                min_val, max_val, mean = params
                df = df.withColumn(col, (F.col(col) - mean) / (max_val - min_val) if max_val != min_val else F.lit(0.0))
        
        return df

class PandasNormalizer(BaseNormalizer):
    """Pandas DataFrame normalization implementation."""
    
    def fit(self, df: pd.DataFrame) -> 'PandasNormalizer':
        """Compute normalization statistics from training data."""
        missing = [col for col in self.columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        for col, method in self.methods.items():
            if method == 'minmax':
                self.stats[col] = {'method': method, 'min': df[col].min(), 'max': df[col].max()}
            elif method == 'zscore':
                self.stats[col] = {'method': method, 'mean': df[col].mean(), 'std': df[col].std()}
            elif method == 'range':
                self.stats[col] = {'method': method, 'min': df[col].min(), 'max': df[col].max(), 'mean': df[col].mean()}
        
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply normalization to DataFrame."""
        if not self.stats:
            raise RuntimeError("Fit the normalizer first")

        df = df.copy()
        for col in self.columns:
            stat = self.stats[col]
            method = stat['method']
            
            if method == 'minmax':
                min_val, max_val = stat['min'], stat['max']
                df[col] = 0.0 if max_val == min_val else (df[col] - min_val) / (max_val - min_val)
            elif method == 'zscore':
                mean, std = stat['mean'], stat['std'] or 1e-19
                df[col] = (df[col] - mean) / std
            elif method == 'range':
                min_val, max_val, mean = stat['min'], stat['max'], stat['mean']
                df[col] = 0.0 if max_val == min_val else (df[col] - mean) / (max_val - min_val)
        
        return df