# Файл: main.py
import sys
from pyspark.sql import SparkSession
from config.registries import MODEL_REGISTRY, METRIC_REGISTRY
from config.validator import load_config
from cli.parsers import create_root_parser, create_method_subparsers
from core.factory import factory

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

def main():
    # Initialize command line interface
    parser = create_root_parser()
    create_method_subparsers(parser)
    args = parser.parse_args()

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
    components = {
        'normalizer': factory.create_normalizer(spark = spark, **config.get('preprocessing').get('normalizer')),
        'sampler': factory.create_sampler(spark = spark, **config.get('preprocessing').get('sampler'))
    }

    # Initialize core components
    model_class = MODEL_REGISTRY[config['algorithm']]['class']
    data_loader = factory.create_loader(
        config['data_path'],
        spark=spark,
        **components
    )
    
    # Execute optimization pipeline
    optimizer = factory.create_optimizer(config.get('optimizer', 'grid'))
    metric = factory.create_metric(config['metric'])
    
    best_params = optimizer.find_best(
        model_class=model_class,
        data_loader=data_loader,
        param_grid=config['params'],
        metric=metric
    )

    # Save final model if requested
    if output_path := config.get('output_path'):
        best_model = factory.create_model(config['algorithm'], best_params)
        best_model.fit(data_loader)
        best_model.save(output_path)

    print(f"Optimal parameters: {best_params}")

if __name__ == "__main__":
    main()