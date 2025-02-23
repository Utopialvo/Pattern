# Файл: core/interfaces.py
from abc import ABC, abstractmethod
from typing import Iterator, Optional, Dict, Any
import pandas as pd


class DataLoader(ABC):
    """Абстрактный класс для загрузки данных батчами."""
    
    @abstractmethod
    def iter_batches(self) -> Iterator[pd.DataFrame]:
        """Генератор батчей данных.
        
        Yields:
            pd.DataFrame: Очередной батч данных
        """
        pass
    
    @abstractmethod
    def full_data(self) -> Optional[pd.DataFrame]:
        """Полная загрузка данных в память.
        
        Returns:
            pd.DataFrame | None: Полный датасет или None если не поддерживается
        """
        pass


class ClusterModel(ABC):
    """Абстрактный класс модели кластеризации."""
    
    @abstractmethod
    def __init__(self, params: Dict[str, Any]):
        """Инициализация модели с параметрами.
        
        Args:
            params (dict): Словарь гиперпараметров модели
        """
        pass
    
    @abstractmethod
    def fit(self, data_loader: DataLoader) -> None:
        """Обучение модели на данных из DataLoader.
        
        Args:
            data_loader (DataLoader): Источник данных для обучения
        """
        pass
    
    @abstractmethod
    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Предсказание кластеров для новых данных.
        
        Args:
            data (pd.DataFrame): Данные для предсказания
            
        Returns:
            pd.Series: Метки кластеров
        """
        pass
    
    @property
    @abstractmethod
    def model_data(self) -> dict:
        """Дополнительные данные модели (центроиды и т.д.).
        
        Returns:
            dict: Словарь с внутренними данными модели
        """
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, path: str) -> 'ClusterModel':
        """Загрузка модели из файла.
        
        Args:
            path (str): Путь к файлу модели
            
        Returns:
            ClusterModel: Загруженная модель
        """
        pass
    
    @abstractmethod
    def save(self, path: str):
        """Сохранение модели в файл.
        
        Args:
            path (str): Путь для сохранения файла
        """
        pass

class Metric(ABC):
    """Абстрактный класс для метрик оценки качества."""
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame, labels: pd.Series, model_data: dict) -> float:
        """Вычисление значения метрики.
        
        Args:
            data (pd.DataFrame): Исходные данные
            labels (pd.Series): Предсказанные метки кластеров
            model_data (dict): Дополнительные данные модели
            
        Returns:
            float: Значение метрики
        """
        pass

class Optimizer(ABC):
    """Абстрактный класс для оптимизации гиперпараметров."""
    
    @abstractmethod
    def find_best(self, model_class: type, data_loader: DataLoader, 
                 param_grid: Dict[str, list], metric: Metric) -> Dict[str, Any]:
        """Поиск наилучших гиперпараметров.
        
        Args:
            model_class (type): Класс модели для оптимизации
            data_loader (DataLoader): Источник данных
            param_grid (dict): Сетка параметров для поиска
            metric (Metric): Метрика для оценки
            
        Returns:
            dict: Наилучшие найденные параметры
        """
        pass