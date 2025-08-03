# Файл: optimization/strategies.py
from typing import Dict, Any, Type, List
import optuna
from optuna.samplers import TPESampler, RandomSampler, GridSampler
from pattern.core.interfaces import Optimizer, DataLoader, Metric

from pattern.core.logger import log_errors


class BaseOptimizer(Optimizer):
    """Base class for optimization"""
    
    def __init__(self, 
                 sampler: str = 'tpe', 
                 direction: str = 'maximize'):
        """
        Args:
            sampler: Optuna sampler type ('tpe', 'random', 'grid')
            direction: Optimization direction (our case 'maximize', becouse SW or GAP 1 is best, and ANUI or modularity 1 is best)
        """
        self.sampler = sampler
        self.direction = direction
        self.param_grid = None
        
    def _get_sampler(self):
        if self.sampler == 'grid':
            if not self.param_grid:
                raise ValueError("requires param_grid")
            return GridSampler(self.param_grid, seed=0)
        elif self.sampler == 'tpe':
            return TPESampler(seed=0)
        elif self.sampler == 'random':
            return RandomSampler(seed=0)
    
    def _calculate_n_trials(self, param_grid: Dict[str, List[Any]]) -> int:
        """Automatically determine optimal number of trials."""
        total_combinations = 1
        for values in param_grid.values():
            total_combinations *= len(values) # заглушка чтобы не использовать itertools
            
        if isinstance(self.sampler, GridSampler):
            return total_combinations
        if total_combinations <= 50:
            return total_combinations
        auto_trials = max(50, int(0.33 * total_combinations)) # 33% от общего заглушка
        return auto_trials

    def find_best(self, 
                 model_class: Type, 
                 data_loader: DataLoader, 
                 param_grid: Dict[str, list], 
                 metric: Metric) -> Dict[str, Any]:
        
        self.param_grid = param_grid
        sampler = self._get_sampler()
        n_trials = self._calculate_n_trials(param_grid)
        
        if isinstance(self.sampler, GridSampler):
            study = optuna.create_study(
                direction=self.direction,
                sampler=sampler,
                search_space=self.param_grid
            )
        else:
            study = optuna.create_study(
                direction=self.direction,
                sampler=sampler
            )
            
        @log_errors
        def objective(trial):
            params = {}
            for param, values in param_grid.items():
                if all(isinstance(v, (int, float)) for v in values):
                    if all(isinstance(v, int) for v in values):
                        params[param] = trial.suggest_int(param, min(values), max(values))
                    else:
                        params[param] = trial.suggest_float(param, min(values), max(values))
                else:
                    params[param] = trial.suggest_categorical(param, values)
            
            try:
                model = model_class(params)
                model.fit(data_loader)
                # labels = model.predict(data_loader)
                labels = model.labels_
                score = metric.calculate(data_loader, labels, model.model_data)
                return score
            except Exception as e:
                raise optuna.TrialPruned()

        study.optimize(objective, n_trials=n_trials)
        return study.best_params

class TPESearch(BaseOptimizer):
    """TPE optimization"""
    def __init__(self, n_trials=100):
        super().__init__(sampler='tpe')

class RandomSearch(BaseOptimizer):
    """Random search optimization"""
    def __init__(self, n_trials=100):
        super().__init__(sampler='random')

class GridSearch(BaseOptimizer):
    """Grid search optimization"""
    def __init__(self):
        super().__init__(sampler='grid') 