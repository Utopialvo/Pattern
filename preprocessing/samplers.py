# Файл: preprocessing/samplers.py
from abc import ABC, abstractmethod
from typing import Union, List, Optional, Tuple
import pandas as pd
from pyspark.sql import SparkSession, DataFrame as SparkDF
import os


class BaseSampler(ABC):
    """Base class for data sampling strategies.
    
    Attributes:
        data_src (Union[str, List[str]]): Path(s) to the data source(s).
        features_sample (Optional[Union[pd.DataFrame, SparkDF]]): Sampled features.
        similarity_sample (Optional[Union[pd.DataFrame, SparkDF]]): Sampled similarity matrix.
    """
    
    def __init__(self, data_src: Union[str, List[str]]):
        self.data_src = data_src
        self.features_sample = None
        self.similarity_sample = None
        self._load()

    def _load(self) -> None:
        """Load existing samples or prepare for new sampling"""
        if isinstance(self.data_src, list) and len(self.data_src) > 2:
            raise ValueError("Maximum 2 data sources supported")
            
        sources = self.data_src if isinstance(self.data_src, list) else [self.data_src]
        
        self.features_sample = self._check(sources[0]) if len(sources) >= 1 else None
        if len(sources) >= 2:
            self.similarity_sample = self._check(sources[1])

    def _check(self, source: str) -> Optional[Union[pd.DataFrame, SparkDF]]:
        """Check for existing samples and load if available"""
        if not isinstance(source, str) or len(source.strip()) == 0:
            return None
        dot_count = source.count('.')
        # if dot_count == 1:
        #     part1, part2 = source.split('.', 1)
        if dot_count == 2:
            if source.split('.')[-2] in ['sample', 'sample_similarity']:
                if self._file_exists(sample_path):
                    return self._load_sample(sample_path)
            else:
                return None
        return None

    @abstractmethod
    def _file_exists(self, path: str) -> bool:
        """Check if sample file exists"""
        pass

    @abstractmethod
    def _load_sample(self, path: str) -> Union[pd.DataFrame, SparkDF]:
        """Load sample data from storage"""
        pass

    @abstractmethod
    def _takesample(self, features: Union[pd.DataFrame, SparkDF], 
                   similarity: Optional[Union[pd.DataFrame, SparkDF]]) -> None:
        """placeholder"""
        pass

    @abstractmethod
    def _savesample(self) -> None:
        """Save generated samples to storage"""
        pass

    def transform(self, 
                 features: Union[pd.DataFrame, SparkDF],
                 similarity: Optional[Union[pd.DataFrame, SparkDF]]) -> Tuple[Union[pd.DataFrame, SparkDF], Optional[Union[pd.DataFrame, SparkDF]]]:
        """Main interface for sampling workflow"""
        if self.features_sample is not None and self.similarity_sample is not None:
            return self.features_sample, self.similarity_sample
            
        self._takesample(features, similarity)
        self._savesample()
        return self.features_sample, self.similarity_sample



class PandasSampler(BaseSampler):
    """Sampling implementation for Pandas DataFrames"""
    
    def __init__(self, data_src: Union[str, List[str]]):
        super().__init__(data_src)
    
    def _file_exists(self, path: str) -> bool:
        return os.path.exists(path)

    def _load_sample(self, path: str) -> pd.DataFrame:
        try:
            if path.endswith('.parquet'):
                return pd.read_parquet(path)
            return pd.read_csv(path)
        except Exception as e:
            raise IOError(f"Failed to load sample: {str(e)}")

    def _takesample(self, 
                   features: pd.DataFrame, 
                   similarity: Optional[pd.DataFrame]) -> None:
        """placeholder"""
        self.features_sample = features.sample(frac=0.1, random_state=42)
        if similarity is not None:
            self.similarity_sample = similarity.sample(frac=0.1, random_state=42)

    def _savesample(self) -> None:
        """Save samples with .sample suffix"""
        if self.features_sample is not None:
            base_path = self.data_src[0] if isinstance(self.data_src, list) else self.data_src
            base, ext = os.path.splitext(base_path)
            save_path = f"{base}.sample{ext}"
            
            self.features_sample.to_parquet(save_path)
            if self.similarity_sample is not None:
                sim_save_path = f"{base}.sample_similarity{ext}"
                self.similarity_sample.to_parquet(sim_save_path)



class SparkSampler(BaseSampler):
    """Sampling implementation for Spark DataFrames"""
    
    def __init__(self, data_src: Union[str, List[str]], spark: SparkSession):
        self.spark = spark
        super().__init__(data_src)

    def _file_exists(self, path: str) -> bool:
        try:
            fs = self.spark._jvm.org.apache.hadoop.fs.FileSystem.get(
                self.spark._jvm.org.apache.hadoop.conf.Configuration())
            return fs.exists(self.spark._jvm.org.apache.hadoop.fs.Path(path))
        except Exception as e:
            raise IOError(f"File check failed: {str(e)}")

    def _load_sample(self, path: str) -> SparkDF:
        try:
            if path.endswith('.parquet'):
                return self.spark.read.parquet(path)
            return self.spark.read.csv(path, header=True)
        except Exception as e:
            raise IOError(f"Failed to load sample: {str(e)}")

    def _takesample(self, 
                   features: SparkDF, 
                   similarity: Optional[SparkDF]) -> None:
        """placeholder"""
        self.features_sample = features.limit(1000)
        if similarity is not None:
            self.similarity_sample = similarity.limit(1000)

    def _savesample(self) -> None:
        """Save samples with .sample suffix"""
        if self.features_sample is not None:
            base_path = self.data_src[0] if isinstance(self.data_src, list) else self.data_src
            base, ext = os.path.splitext(base_path)
            save_path = f"{base}.sample{ext}"
            
            self.features_sample.write.parquet(save_path, mode='overwrite')
            if self.similarity_sample is not None:
                sim_save_path = f"{base}_similarity.sample{ext}"
                self.similarity_sample.write.parquet(sim_save_path, mode='overwrite')