# Файл: config/validator.py
import json
from typing import Dict
from config.registries import MODEL_REGISTRY, METRIC_REGISTRY

def validate_config(config: Dict):
    """Проверяет корректность конфигурации.
    
    Args:
        config (dict): Загруженный конфигурационный файл
        
    Raises:
        ValueError: При обнаружении ошибок в конфигурации
    """
    required = {'data_source', 'data_path', 'algorithm', 'params', 'metric'}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Missing config fields: {missing}")
    
    if config['algorithm'] not in MODEL_REGISTRY:
        raise ValueError(f"Invalid algorithm: {config['algorithm']}")
    
    if config['metric'] not in METRIC_REGISTRY:
        raise ValueError(f"Invalid metric: {config['metric']}")

def load_config(path: str) -> Dict:
    """Загрузка и валидация конфигурационного файла.
    
    Args:
        path (str): Путь к JSON-файлу конфигурации
        
    Returns:
        dict: Валидный конфигурационный словарь
    """
    with open(path) as f:
        config = json.load(f)
    validate_config(config)
    return config