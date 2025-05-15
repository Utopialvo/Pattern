# Файл: cli/parsers.py
import argparse
from typing import Dict, Any
from config.registries import MODEL_REGISTRY, METRIC_REGISTRY

def create_root_parser() -> argparse.ArgumentParser:
    """Create the root argument parser with common parameters."""
    parser = argparse.ArgumentParser(
        prog='main.py',
        description='Pattern',
        add_help=False,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    main_group = parser.add_argument_group('Main parameters')
    main_group.add_argument(
        '-c', '--config_path',
        nargs='?',
        help='Path to configuration file'
    )
    
    help_group = parser.add_argument_group('Help')
    help_group.add_argument(
        '-h', '--help',
        action='store_true',
        help='Show this help message and exit'
    )
    help_group.add_argument(
        '-l', '--list',
        action='store_true',
        help='List available algorithms and metrics'
    )
    return parser

def create_method_subparsers(parser: argparse.ArgumentParser) -> None:
    """Add algorithm-specific subparsers to the root parser."""
    subparsers = parser.add_subparsers(
        title='Available algorithms',
        dest='algorithm',
        help='Use main.py <algorithm> -h for specific help'
    )
    
    for algo_name, algo_info in MODEL_REGISTRY.items():
        algo_parser = subparsers.add_parser(
            algo_name,
            help=f"{algo_name} algorithm parameters",
            formatter_class=argparse.RawTextHelpFormatter
        )
        _add_algorithm_params(algo_parser, algo_info)

def _add_algorithm_params(parser: argparse.ArgumentParser, algo_info: Dict[str, Any]) -> None:
    """Add algorithm-specific parameters to a subparser."""
    params_group = parser.add_argument_group('Algorithm parameters')
    for param, desc in algo_info['params_help'].items():
        is_required = 'required' in desc.lower()
        help_msg = f"{desc} (required)" if is_required else desc
        params_group.add_argument(
            f'--{param}',
            type=_auto_type(desc),
            required=is_required,
            help=help_msg
        )

def _auto_type(desc: str) -> type:
    """Infer parameter type from its description."""
    desc_lower = desc.lower()
    if 'positive integer' in desc_lower:
        return int
    if 'positive float' in desc_lower:
        return float
    return str