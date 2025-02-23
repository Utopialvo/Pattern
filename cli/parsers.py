# Файл: cli/parsers.py
import argparse
from typing import Dict, Any
from config.registries import MODEL_REGISTRY, METRIC_REGISTRY

def create_root_parser() -> argparse.ArgumentParser:
    """Создает корневой парсер аргументов."""
    parser = argparse.ArgumentParser(
        prog='main.py',
        description='Cluster Analysis Toolkit',
        add_help=False,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    main_group = parser.add_argument_group('Основные параметры')
    main_group.add_argument(
        'config_path',
        nargs='?',
        help='Путь к конфигурационному файлу'
    )
    
    help_group = parser.add_argument_group('Справка')
    help_group.add_argument(
        '-h', '--help',
        action='store_true',
        help='Показать общую справку'
    )
    help_group.add_argument(
        '-l', '--list',
        action='store_true',
        help='Список доступных алгоритмов и метрик'
    )
    
    return parser

def create_method_subparsers(parser: argparse.ArgumentParser):
    """Добавляет подпарсеры для каждого алгоритма."""
    subparsers = parser.add_subparsers(
        title='Доступные алгоритмы',
        dest='algorithm',
        help='Используйте main.py <алгоритм> -h для получения справки'
    )
    
    for algo_name, algo_info in MODEL_REGISTRY.items():
        algo_parser = subparsers.add_parser(
            algo_name,
            help=f"Параметры алгоритма {algo_name}",
            formatter_class=argparse.RawTextHelpFormatter
        )
        _add_algorithm_params(algo_parser, algo_info)

def _add_algorithm_params(parser: argparse.ArgumentParser, algo_info: Dict[str, Any]):
    """Добавляет параметры алгоритма в парсер."""
    params_group = parser.add_argument_group('Параметры алгоритма')
    for param, desc in algo_info['params_help'].items():
        params_group.add_argument(
            f'--{param}',
            type=_auto_type(desc),
            help=f"{desc} (обязательный)" if 'required' in desc.lower() else desc
        )

def _auto_type(desc: str) -> type:
    """Определяет тип параметра по описанию."""
    if 'positive integer' in desc:
        return int
    if 'positive float' in desc:
        return float
    if '[' in desc and ']' in desc:
        return str
    return str