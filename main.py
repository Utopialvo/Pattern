# Файл: main.py
import sys
import json
import pandas as pd
from pyspark.sql import SparkSession
from config.registries import MODEL_REGISTRY, METRIC_REGISTRY
from config.validator import load_config
from cli.parsers import create_root_parser, create_method_subparsers
from core.factory import factory 


def print_help():
    """Выводит расширенную справку."""
    help_text = f"""
Доступные алгоритмы ({len(MODEL_REGISTRY)}):
{', '.join(MODEL_REGISTRY.keys())}

Доступные метрики ({len(METRIC_REGISTRY)}):
{', '.join(METRIC_REGISTRY.keys())}

Примеры использования:
1. Запуск с конфигурационным файлом:
   main.py config.json

2. Справка по алгоритму:
   main.py kmeans -h
"""
    print(help_text)

def handle_list_command():
    """Обрабатывает запрос списка алгоритмов."""
    print("Реализованные алгоритмы:")
    for algo, info in MODEL_REGISTRY.items():
        print(f"\n{algo}:")
        print(f"  Параметры: {', '.join(info['params_help'].keys())}")
    
    print("\nДоступные метрики:")
    print('\n'.join(METRIC_REGISTRY.keys()))

def main():
    parser = create_root_parser()
    create_method_subparsers(parser)
    args = parser.parse_args()
    
    if args.help:
        print_help()
        return
    
    if args.list:
        handle_list_command()
        return
    
    if args.algorithm:
        # Обработка вызова для конкретного алгоритма
        return
    
    if not args.config_path:
        print("Ошибка: Не указан конфигурационный файл")
        sys.exit(1)
    
    # Загрузка и выполнение конфигурации
    config = load_config(args.config_path)
    
    # Инициализация компонентов
    if config['data_source'] == 'pandas':
        data = pd.read_parquet(config['data_path'])
        data_loader = factory.create_loader(config['data_path'])
    elif config['data_source'] == 'spark':
        spark = SparkSession.builder.getOrCreate()
        data_loader = factory.create_loader(config['data_path'], spark = spark)
    else:
        raise ValueError(f"Неподдерживаемый источник данных: {config['data_source']}")
    
    model_class = MODEL_REGISTRY[config['algorithm']]['class']
    optimizer = factory.create_optimizer(config.get('optimizer', 'grid'))
    metric = factory.create_metric(config['metric'])
    
    # Запуск оптимизации
    best_params = optimizer.find_best(
        model_class=model_class,
        data_loader=data_loader,
        param_grid=config['params'],
        metric=metric
    )
    
    # Создание финальной модели с лучшими параметрами
    if 'output_path' in config:
        best_model = factory.create_model(config['algorithm'], best_params)
        if config['data_source'] == 'pandas':
            data = pd.read_parquet(config['data_path'])
            data_loader = factory.create_loader(config['data_path'])
        elif config['data_source'] == 'spark':
            data_loader = factory.create_loader(config['data_path'], spark = spark)
        best_model.fit(data_loader = data_loader)
        best_model.save(config['output_path'])
    
    print(f"Оптимальные параметры: {best_params}")

if __name__ == "__main__":
    main()