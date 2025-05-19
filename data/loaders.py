# Файл: data/loaders.py
from typing import Iterator, Optional, Union, Tuple, Any
import pandas as pd
from pyspark.sql import SparkSession, DataFrame as SparkDataFrame
from core.interfaces import DataLoader
from abc import abstractmethod
from core.logger import logger
from data.utils import transform_edgelist

class BaseDataLoader(DataLoader):
    """Base data loader implementing common functionality."""
    
    def __init__(self, 
                 features: Optional[Union[str, pd.DataFrame, SparkDataFrame]] = None, 
                 similarity: Optional[Union[str, pd.DataFrame, SparkDataFrame]] = None,
                 normalizer: Optional[Any] = None,
                 sampler: Optional[Any] = None,
                 **kwargs):
        self.normalizer = normalizer
        self.sampler = sampler
        self.features = None
        self.similarity_matrix = None
        self._load(features, similarity)

    def _load(self, 
                features: Optional[Union[str, pd.DataFrame, SparkDataFrame]], 
                similarity: Optional[Union[str, pd.DataFrame, SparkDataFrame]]) -> None:
        """Load and preprocess data from source(s)."""
        if not isinstance(features, type(None)):
            self.features = self._load_source(features)
        if not isinstance(similarity, type(None)):
            self.similarity_matrix = self._load_source(similarity)

        if self.sampler:
            self.features, self.similarity_matrix = self.sampler.transform(
                self.features, self.similarity_matrix
            )

        if self.normalizer and self.features is not None:
            self.normalizer.fit(self.features)
            if isinstance(features, str):
                self.normalizer.save(f"{features.split('.')[0]}.normstats.joblib")
            self.features = self.normalizer.transform(self.features)

        self.similarity_matrix = transform_edgelist(self.similarity_matrix)
        
    def full_data(self) -> Tuple:
        """Return complete dataset as (features, similarity_matrix)."""
        return (self.features, self.similarity_matrix)

    @abstractmethod
    def _load_source(self, source: Union[str, Any]):
        """Load individual data source (implemented in subclasses)."""

    @abstractmethod
    def iter_batches(self, batch_size: int) -> Iterator[Tuple]:
        pass

class PandasDataLoader(BaseDataLoader):
    """Data loader for pandas DataFrames."""
    
    def _load_source(self, source: Union[str, pd.DataFrame]) -> pd.DataFrame:
        if isinstance(source, str):
            try:
                logger.info(f"Loading data from {source}")
                return pd.read_parquet(source) if source.endswith('.parquet') else pd.read_csv(source)
            except Exception as e:
                logger.error(f"Failed to load {source}: {str(e)}")
                raise
        return source

    def iter_batches(self, batch_size: int) -> Iterator[Tuple]:
        """Batch data generator (implementation placeholder)."""
        pass

class SparkDataLoader(BaseDataLoader):
    """Data loader for Spark DataFrames."""
    
    def __init__(self, 
                 spark: SparkSession,
                 *args, 
                 **kwargs):
        self.spark = spark
        super().__init__(*args, **kwargs)

    def _load_source(self, source: Union[str, SparkDataFrame]) -> SparkDataFrame:
        format_loaders = {
            'parquet': self.spark.read.parquet,
            'orc': self.spark.read.orc,
            'csv': lambda x: self.spark.read.csv(x, header=True)
        }
        if isinstance(source, str):
            try:
                logger.info(f"Loading data from {source}")
                fmt = source.split('.')[-1]
                return format_loaders.get(fmt, format_loaders['parquet'])(source)
            except Exception as e:
                logger.error(f"Failed to load {source}: {str(e)}")
                raise
        return source

    def iter_batches(self, batch_size: int) -> Iterator[Tuple]:
        """Batch data generator (implementation placeholder)."""
        pass