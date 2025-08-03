# Файл: config/validator.py
import json
from typing import Dict, Any
from pattern.config.registries import MODEL_REGISTRY, METRIC_REGISTRY

def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration structure and content.
    
    Args:
        config: Configuration dictionary to validate
        
    Raises:
        ValueError: For any configuration integrity issues
    """    
    if config['algorithm'] not in MODEL_REGISTRY:
        raise ValueError(f"Unregistered algorithm: {config['algorithm']}")
    
    if config['metric'] not in METRIC_REGISTRY:
        raise ValueError(f"Unregistered metric: {config['metric']}")

    required = {'data_source', 'features', 'similarity', 'algorithm', 'params', 'metric'}
    if missing := required - config.keys():
        raise ValueError(f"Missing required fields: {missing}")


def load_config(path: str) -> Dict[str, Any]:
    """Load and validate configuration from JSON file.
    
    Args:
        path: Path to JSON configuration file
        
    Returns:
        Validated configuration dictionary
        
    Raises:
        FileNotFoundError: If file doesn't exist
        JSONDecodeError: If invalid JSON format
        ValueError: If configuration validation fails
    """
    with open(path) as f:
        config = json.load(f)
    validate_config(config)
    return config