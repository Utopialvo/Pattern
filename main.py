# Файл: main.py
import sys
import logging
from pyspark.sql import SparkSession
from config.registries import MODEL_REGISTRY, METRIC_REGISTRY
from config.validator import load_config
from cli.parsers import create_root_parser, create_method_subparsers
from core.factory import factory
from core.logger import logger, log_errors


def print_help():
    """Display extended help information."""
    help_text = f"""
Available algorithms ({len(MODEL_REGISTRY)}):
{', '.join(MODEL_REGISTRY.keys())}

Available metrics ({len(METRIC_REGISTRY)}):
{', '.join(METRIC_REGISTRY.keys())}

Usage examples:
1. Run with config file:
   main.py config.json

2. Algorithm help:
   main.py kmeans -h
"""
    print(help_text)

def handle_list_command():
    """Display list of available algorithms and metrics."""
    print("Implemented algorithms:")
    for algo, info in MODEL_REGISTRY.items():
        params = ', '.join(info['params_help'].keys())
        print(f"\n{algo}:\n  Parameters: {params}")
    
    print("\nAvailable metrics:")
    print('\n'.join(METRIC_REGISTRY.keys()))

@log_errors
def main():
    # Initialize command line interface
    parser = create_root_parser()
    create_method_subparsers(parser)
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    if args.help:
        print_help()
        return
    
    if args.list:
        handle_list_command()
        return

    if not args.config_path:
        sys.exit("Error: Configuration file not specified")

    # Load and validate configuration
    config = load_config(args.config_path)
    
    # Initialize execution environment
    spark = SparkSession.builder.getOrCreate() if config['data_source'] == 'spark' else None
    
    # Configure data processing components
    if sampler := config.get('preprocessing').get('sampler'):
        sampler = factory.create_sampler(spark = spark, **sampler)
    if normalizer := config.get('preprocessing').get('normalizer'):
        normalizer = factory.create_normalizer(spark = spark, **normalizer)

    # Initialize core components
    model_class = MODEL_REGISTRY[config['algorithm']]['class']
    data_loader = factory.create_loader(
        config['data_path'],
        spark=spark,
        normalizer = normalizer,
        sampler = sampler
    )
    
    # Execute optimization pipeline
    optimizer = factory.create_optimizer(config.get('optimizer', 'grid'))
    metric = factory.create_metric(config['metric'])

    print('Start find best params...')
    best_params = optimizer.find_best(
        model_class=model_class,
        data_loader=data_loader,
        param_grid=config['params'],
        metric=metric
    )
    print(f"Optimal parameters: {best_params}")

    
    # Save final model if requested
    if output_path := config.get('output_path'):
        best_model = factory.create_model(config['algorithm'], best_params)
        best_model.fit(data_loader)
        best_model.save(output_path)
        print(f"Saving model: {output_path}")

    # Visualize result model
    if plots_path := config.get('plots_path'):
        visualizer = factory.create_visualizer(plots_path)
        visualizer.visualisation(data_loader, best_model.labels_)


if __name__ == "__main__":
    main()