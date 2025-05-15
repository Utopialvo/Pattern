# Файл: optimization/strategies.py
import itertools
import random
from typing import Dict, Any, Type
from core.interfaces import Optimizer, DataLoader, Metric

class GridSearch(Optimizer):
    """Exhaustive search over hyperparameter combinations."""
    
    def find_best(self, 
                 model_class: Type, 
                 data_loader: DataLoader, 
                 param_grid: Dict[str, list],
                 metric: Metric) -> Dict[str, Any]:
        """Perform grid search optimization.
        
        Args:
            model_class: Model class to optimize
            data_loader: Data source for training/validation
            param_grid: Parameter search space {parameter: [values]}
            metric: Evaluation metric to maximize
            
        Returns:
            Best performing parameter configuration
            
        Note:
            Prints information about failed parameter combinations
        """
        best_score = -float('inf')
        best_params = {}
        
        for params in self._generate_params(param_grid):
            try:
                model = model_class(params)
                model.fit(data_loader)
                labels = model.predict(data_loader)
                
                score = metric.calculate(
                    data_loader=data_loader,
                    labels=labels,
                    model_data=model.model_data
                )
                
                if score > best_score:
                    best_score, best_params = score, params.copy()
                    
            except Exception as e:
                print(f"Skipped {params}: {str(e)}")
                
        return best_params

    def _generate_params(self, param_grid: Dict[str, list]) -> Dict[str, Any]:
        """Generate all parameter combinations from grid."""
        for values in itertools.product(*param_grid.values()):
            yield dict(zip(param_grid.keys(), values))

class RandomSearch(GridSearch):
    """Random parameter search with limited iterations."""
    
    def __init__(self, n_iter: int = 10):
        """
        Args:
            n_iter: Number of random parameter combinations to test
        """
        self.n_iter = n_iter

    def _generate_params(self, param_grid: Dict[str, list]) -> Dict[str, Any]:
        """Generate random parameter combinations."""
        for _ in range(self.n_iter):
            yield {k: random.choice(v) for k, v in param_grid.items()}