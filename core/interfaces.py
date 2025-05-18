# Файл: core/interfaces.py
from abc import ABC, abstractmethod
from typing import Iterator, Optional, Dict, Any, Tuple, Union
import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame


class DataLoader(ABC):
    """Abstract base class for data loading components."""
    
    @abstractmethod
    def __init__(self, 
                 data_src: Union[str, list, Tuple],
                 normalizer: Optional['Normalizer'] = None,
                 sampler: Optional['Sampler'] = None):
        """
        Initialize data loader with source and processing components
        
        Args:
            data_src: Data source(s) - path(s) or DataFrame(s)
            normalizer: Optional data normalization component
            sampler: Optional data sampling component
        """

    @abstractmethod
    def _load(self, *data_src: Union[str, pd.DataFrame, SparkDataFrame]):
        """Load and preprocess data from source(s)."""

    @abstractmethod
    def iter_batches(self, batch_size: int) -> Iterator[Tuple]:
        """Generate data batches with optional shuffling."""

    @abstractmethod
    def full_data(self) -> Tuple:
        """Return complete dataset as (features, similarity_matrix)."""


class ClusterModel(ABC):
    """Abstract base class for clustering model implementations."""
    
    @abstractmethod
    def __init__(self, params: Dict[str, Any]):
        """
        Initialize model with configuration parameters
        
        Args:
            params: Model hyperparameters dictionary
        """
    
    @abstractmethod
    def fit(self, data_loader: DataLoader) -> None:
        """Train model using data from provided loader"""
    
    @abstractmethod
    def predict(self, data_loader: DataLoader) -> Union[pd.Series, pd.DataFrame, SparkDataFrame]:
        """Generate cluster predictions for input data"""
    
    @property
    @abstractmethod
    def model_data(self) -> dict:
        """Access internal model state data (centroids, etc.)"""
    
    @classmethod
    @abstractmethod
    def load(cls, path: str) -> 'ClusterModel':
        """Reconstruct model from storage"""
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Persist model to storage"""

class Metric(ABC):
    """Abstract base class for clustering evaluation metrics."""
    
    @abstractmethod
    def calculate(self,
                  data_loader: DataLoader,
                  labels: Union[pd.Series, pd.DataFrame, SparkDataFrame], 
                  model_data: dict) -> float:
        """
        Compute metric value for clustering results
        
        Args:
            data_loader: Source of input data
            labels: Predicted cluster assignments
            model_data: Internal model state data
            
        Returns:
            Computed metric score
        """

class Optimizer(ABC):
    """Abstract base class for hyperparameter optimization strategies."""
    
    @abstractmethod
    def find_best(self, 
                  model_class: type[ClusterModel], 
                  data_loader: DataLoader, 
                  param_grid: Dict[str, list], 
                  metric: Metric) -> Dict[str, Any]:
        """
        Discover optimal hyperparameters through search
        
        Args:
            model_class: Model type to optimize
            data_loader: Training data source
            param_grid: Parameter search space
            metric: Optimization target metric
            
        Returns:
            Best performing parameter configuration
        """

class DataVis(ABC):
    """Abstract base class for clustering visualisation."""
    
    @abstractmethod
    def visualisation(self,
                  data_loader: DataLoader,
                  labels: Union[pd.Series, pd.DataFrame, SparkDataFrame], 
                  model_data: dict) -> None:
        """
        Create visualisation for clustering results
        
        Args:
            data_loader: Source of input data
            labels: Predicted cluster assignments
            model_data: Internal model state data
            
        Returns:
            Save clustering visualisation
        """

class DataStatistics(ABC):
    """Abstract base class for clustering visualisation."""
    
    @abstractmethod
    def compute_statistics(self,
                  data_loader: DataLoader,
                  labels: Union[pd.Series, pd.DataFrame, SparkDataFrame], 
                  alpha: float = 0.05) -> None:
        """
        Compute and save statistics clustering
        
        Args:
            data_loader: Source of input data
            labels: Predicted cluster assignments
            alpha: significant level
        """

class Normalizer(ABC):
    """Abstract base class for data normalization."""
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Persist normalization parameters to disk."""        
    @abstractmethod
    def load(self, path: str) -> None:
        """Load normalization parameters from disk"""
    @abstractmethod
    def fit(self, df: Union[pd.DataFrame, SparkDataFrame]) -> None:
        """Compute normalization statistics from training data."""
    @abstractmethod
    def transform(self, df: Union[pd.DataFrame, SparkDataFrame]) -> Union[pd.DataFrame, SparkDataFrame]:
        """Apply normalization to DataFrame."""


class Sampler(ABC):
    """Abstract base class for data sampling."""
    
    @abstractmethod
    def _file_exists(self, path: str) -> bool:
        """Check if sample file exists"""

    @abstractmethod
    def _load_sample(self, path: str) -> Union[pd.DataFrame, SparkDataFrame]:
        """Load sample data from storage"""

    @abstractmethod
    def _takesample(self, features: Union[pd.DataFrame, SparkDataFrame], 
                   similarity: Optional[Union[pd.DataFrame, SparkDataFrame]]) -> None:
        """placeholder"""

    @abstractmethod
    def _savesample(self) -> None:
        """Save generated samples to storage"""
        
    @abstractmethod
    def transform(self, 
                 features: Union[pd.DataFrame, SparkDataFrame],
                 similarity: Optional[Union[pd.DataFrame, SparkDataFrame]]) -> Tuple[Union[pd.DataFrame, SparkDataFrame], Optional[Union[pd.DataFrame, SparkDataFrame]]]:
        """Main interface for sampling workflow"""


class ComponentFactory(ABC):
    """Abstract factory interface for system component creation."""
    
    @abstractmethod
    def create_model(self, identifier: str, config: Dict[str, Any]) -> ClusterModel:
        """
        Instantiate clustering model
        
        Args:
            identifier: Registered model type identifier
            config: Model initialization parameters
        """
    
    @abstractmethod
    def create_metric(self, identifier: str) -> Metric:
        """Construct evaluation metric instance"""
    
    @abstractmethod
    def create_optimizer(self, identifier: str, **kwargs) -> Optimizer:
        """
        Initialize optimization strategy
        
        Args:
            identifier: Optimizer type identifier
            kwargs: Strategy-specific configuration
        """
    
    @abstractmethod
    def create_loader(self, data_src: Any, **kwargs) -> DataLoader:
        """
        Configure data loading pipeline
        
        Args:
            data_src: Input data source(s)
            kwargs: Loader-specific configuration
        """
        
    @abstractmethod
    def create_sampler(self, data_src: Any, **kwargs) -> Sampler:
        """
        Configure data sampler pipeline
        
        Args:
            data_src: Input data source(s)
            kwargs: sampler-specific configuration
        """
            
    @abstractmethod
    def create_normalizer(self, **kwargs) -> Normalizer:
        """
        Configure data normalizer pipeline
        
        Args:
            kwargs: normalizer-specific configuration
        """
    @abstractmethod
    def create_visualizer(self, plots_path: str) -> DataVis:
         """
        Configure data visualizer pipeline
        
        Args:
            plots_path: Path to save data visualisation
        """
        
    @abstractmethod
    def create_analyser(self, stat_path: str) -> DataStatistics:
         """
        Configure data analyser pipeline
        
        Args:
            stat_path: Path to save report
        """