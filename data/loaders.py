# Файл: data/loaders.py
from typing import Iterator, Optional, Union
import pandas as pd
from pyspark.sql import SparkSession
from core.interfaces import DataLoader
from preprocessing.normalizers import Normalizer

class NormalizingDataLoader(DataLoader):
    """Декоратор для DataLoader с добавлением нормализации."""
    
    def __init__(self, base_loader: DataLoader, normalizer: Normalizer):
        self.base_loader = base_loader
        self.normalizer = normalizer

    def iter_batches(self) -> Iterator[pd.DataFrame]:
        for batch in self.base_loader.iter_batches():
            yield self.normalizer.transform(batch)

    def full_data(self) -> pd.DataFrame:
        return self.normalizer.transform(self.base_loader.full_data())
    

class PandasDataLoader(DataLoader):
    """Загрузчик данных из pandas DataFrame."""
    
    def __init__(self, data: Union[pd.DataFrame, str], batch_size: int = 1000):
        """
        Args:
            data (pd.DataFrame): Исходные данные
            batch_size (int): Размер батча для итерации
        """
        self.data = data
        if isinstance(self.data, str):
            self.data = pd.read_parquet(self.data)
        self.batch_size = batch_size

    def iter_batches(self) -> Iterator[pd.DataFrame]:
        """Итерация по батчам данных.
        
        Examples:
            >>> for batch in loader.iter_batches():
            ...     process(batch)
        """
        for i in range(0, len(self.data), self.batch_size):
            yield self.data.iloc[i:i+self.batch_size]

    def full_data(self) -> pd.DataFrame:
        """Возвращает полный датасет."""
        return self.data

class SparkDataLoader(DataLoader):
    """Загрузчик данных из Spark DataFrame."""
    
    def __init__(self, spark: SparkSession, path: str, format: str = 'parquet'):
        """
        Args:
            spark (SparkSession): Сессия Spark
            path (str): Путь к данным
            format (str): Формат данных (parquet, csv и т.д.)
        """
        """Заглушка"""
        self.spark = spark
        self.path = path
        self.format = format
        self._data = None

    def _load(self):
        """Ленивая загрузка данных при первом обращении."""
        """Заглушка"""
        if self._data is None:
            self._data = self.spark.read.format(self.format).load(self.path)

    def iter_batches(self) -> Iterator[pd.DataFrame]:
        """Итерация по батчам данных через toLocalIterator."""
        """Заглушка"""
        self._load()
        return (batch.toPandas() for batch in self._data.rdd.toLocalIterator())

    def full_data(self) -> Optional[pd.DataFrame]:
        """Заглушка"""
        self._load()
        return self._data.toPandas()