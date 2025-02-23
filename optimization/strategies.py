# Файл: optimization/strategies.py
import itertools
import random
from typing import Dict, Any
from core.interfaces import Optimizer, DataLoader, Metric

class GridSearch(Optimizer):
    """Полный перебор всех комбинаций параметров."""
    
    def find_best(self, model_class: type, data_loader: DataLoader, 
                 param_grid: Dict[str, list], metric: Metric) -> Dict[str, Any]:
        """Поиск по сетке параметров.
        
        Args:
            model_class (type): Класс модели для оптимизации
            data_loader (DataLoader): Источник данных
            param_grid (dict): Сетка параметров вида {параметр: [значения]}
            metric (Metric): Метрика для оценки
            
        Returns:
            dict: Наилучшие параметры
            
        Note:
            Печатает информацию о неудачных комбинациях параметров
        """
        best_score = -float('inf')
        best_params = {}
        
        for params in self._generate_params(param_grid):
            try:
                model = model_class(params)
                model.fit(data_loader)
                labels = model.predict(data_loader.full_data())
                score = metric.calculate(data_loader.full_data(), labels, model.model_data)
                
                if score > best_score:
                    best_score = score
                    best_params = params
            except Exception as e:
                print(f"Skip {params}: {e}")
        
        return best_params

    def _generate_params(self, param_grid):
        """Генератор всех комбинаций параметров."""
        keys, values = zip(*param_grid.items())
        for combination in itertools.product(*values):
            yield dict(zip(keys, combination))

class RandomSearch(GridSearch):
    """Случайный поиск с ограниченным числом итераций."""
    
    def __init__(self, n_iter: int = 10):
        """
        Args:
            n_iter (int): Количество случайных комбинаций для проверки
        """
        self.n_iter = n_iter

    def _generate_params(self, param_grid):
        """Генератор случайных комбинаций параметров."""
        for _ in range(self.n_iter):
            yield {k: random.choice(v) for k, v in param_grid.items()}