# Файл: config/registries.py
import json
import os
import importlib
from typing import Dict, Any, Type

REGISTRY_DIR = "config/registries"
MODEL_REGISTRY_FILE = os.path.join(REGISTRY_DIR, "models_registry.json")
METRIC_REGISTRY_FILE = os.path.join(REGISTRY_DIR, "metrics_registry.json")

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {}
METRIC_REGISTRY: Dict[str, Dict[str, Any]] = {}

def load_registries():
    """Загрузка реестров из JSON-файлов"""
    global MODEL_REGISTRY, METRIC_REGISTRY
    
    if os.path.exists(MODEL_REGISTRY_FILE):
        with open(MODEL_REGISTRY_FILE, 'r') as f:
            MODEL_REGISTRY = json.load(f)
    
    if os.path.exists(METRIC_REGISTRY_FILE):
        with open(METRIC_REGISTRY_FILE, 'r') as f:
            METRIC_REGISTRY = json.load(f)

def register_model(name: str, params_help: Dict[str, str]):
    """Декоратор для регистрации моделей"""
    def decorator(cls: Type) -> Type:
        # Обновление файла реестра
        entry = {
            "module": cls.__module__,
            "class_name": cls.__name__,
            "params_help": params_help
        }
        
        registry = {}
        if os.path.exists(MODEL_REGISTRY_FILE):
            with open(MODEL_REGISTRY_FILE, 'r') as f:
                registry = json.load(f)
        
        registry[name] = entry
        
        os.makedirs(REGISTRY_DIR, exist_ok=True)
        with open(MODEL_REGISTRY_FILE, 'w') as f:
            json.dump(registry, f, indent=2)
        
        return cls
    return decorator

def register_metric(name: str):
    """Декоратор для регистрации метрик"""
    def decorator(cls: Type) -> Type:
        # Обновление файла реестра
        entry = {
            "module": cls.__module__,
            "class_name": cls.__name__
        }
        
        registry = {}
        if os.path.exists(METRIC_REGISTRY_FILE):
            with open(METRIC_REGISTRY_FILE, 'r') as f:
                registry = json.load(f)
        
        registry[name] = entry
        
        os.makedirs(REGISTRY_DIR, exist_ok=True)
        with open(METRIC_REGISTRY_FILE, 'w') as f:
            json.dump(registry, f, indent=2)
        
        return cls
    return decorator

def get_model_class(identifier: str) -> Type:
    """Динамическая загрузка класса модели"""
    if identifier not in MODEL_REGISTRY:
        raise ValueError(f"Model '{identifier}' not found in registry")
    
    entry = MODEL_REGISTRY[identifier]
    module = importlib.import_module(entry["module"])
    return getattr(module, entry["class_name"])

def get_metric_class(identifier: str) -> Type:
    """Динамическая загрузка класса метрики"""
    if identifier not in METRIC_REGISTRY:
        raise ValueError(f"Metric '{identifier}' not found in registry")
    
    entry = METRIC_REGISTRY[identifier]
    module = importlib.import_module(entry["module"])
    return getattr(module, entry["class_name"])

# Инициализация при импорте
load_registries()