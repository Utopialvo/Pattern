# Файл: data/loaders.py
from typing import Iterator, Optional, Union, Tuple, Any
import pandas as pd
from pyspark.sql import SparkSession, DataFrame as SparkDataFrame
from core.interfaces import DataLoader
from abc import abstractmethod

class BaseDataLoader(DataLoader):
    """Base data loader implementing common functionality."""
    
    def __init__(self, 
                 data_src: Union[str, list, tuple, pd.DataFrame, SparkDataFrame],
                 normalizer: Optional[Any] = None,
                 sampler: Optional[Any] = None,
                 **kwargs):
        self.normalizer = normalizer
        self.sampler = sampler
        self._load(data_src)

    def _load(self, data_src: Union[str, list, tuple]) -> None:
        """Load and preprocess data from source(s)."""
        if isinstance(data_src, list) and len(data_src) > 2:
            raise ValueError("Maximum 2 data sources supported")

        sources = data_src if isinstance(data_src, list) else [data_src]
        self.features = self._load_source(sources[0]) if len(sources) >= 1 else None
        self.similarity_matrix = self._load_source(sources[1]) if len(sources) >= 2 else None

        if self.sampler:
            self.features, self.similarity_matrix = self.sampler.transform(
                self.features, self.similarity_matrix
            )

        if self.normalizer and self.features is not None:
            self.features = self.normalizer.transform(self.features)

    def full_data(self) -> Tuple:
        """Return complete dataset as (features, similarity_matrix)."""
        return (self.features, self.similarity_matrix)

    @abstractmethod
    def _load_source(self, source: Union[str, Any]):
        """Load individual data source (implemented in subclasses)."""

class PandasDataLoader(BaseDataLoader):
    """Data loader for pandas DataFrames."""
    
    def _load_source(self, source: Union[str, pd.DataFrame]) -> pd.DataFrame:
        if isinstance(source, str):
            return pd.read_parquet(source) if source.endswith('.parquet') else pd.read_csv(source)
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
            fmt = source.split('.')[-1]
            return format_loaders.get(fmt, format_loaders['parquet'])(source)
        return source

    def iter_batches(self, batch_size: int) -> Iterator[Tuple]:
        """Batch data generator (implementation placeholder)."""
        pass