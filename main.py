# Файл: main.py
import sys
import logging
from contextlib import contextmanager
from enum import Enum
from typing import Optional, Dict, Any
from pyspark.sql import SparkSession
from config.registries import MODEL_REGISTRY, METRIC_REGISTRY
from config.validator import load_config
from cli.parsers import create_root_parser, create_method_subparsers
from core.factory import factory
from core.logger import logger, log_errors


class DataType(Enum):
    """Supported data types for clustering"""
    ATTRIBUTES = "attributes"          # Feature-based data (tabular)
    GRAPH = "graph"                   # Pure network/graph data
    ATTRIBUTED_GRAPH = "attributed_graph"  # Graph with node attributes


class ProcessingMode(Enum):
    """Data processing backends"""
    PANDAS = "pandas"
    SPARK = "spark"


@contextmanager
def get_spark_session(processing_mode: ProcessingMode, spark_config: Optional[Dict[str, Any]] = None):
    """Context manager for Spark session lifecycle management."""
    if processing_mode == ProcessingMode.SPARK:
        builder = SparkSession.builder.appName("Pattern-Clustering")
        
        # Apply custom Spark configuration if provided
        if spark_config:
            for key, value in spark_config.items():
                builder = builder.config(key, value)
        
        spark = builder.getOrCreate()
        logger.info(f"Initialized Spark session: {spark.version}")
        try:
            yield spark
        finally:
            spark.stop()
            logger.info("Spark session terminated")
    else:
        yield None


def validate_data_type_compatibility(config: Dict[str, Any]) -> DataType:
    """Validate and determine data type from configuration."""
    has_features = config.get('features') is not None
    has_graph = config.get('similarity') is not None or config.get('adjacency') is not None
    
    if has_features and has_graph:
        data_type = DataType.ATTRIBUTED_GRAPH
    elif has_graph:
        data_type = DataType.GRAPH
    elif has_features:
        data_type = DataType.ATTRIBUTES
    else:
        raise ValueError("Configuration must specify either 'features', 'similarity'/'adjacency', or both")
    
    logger.info(f"Detected data type: {data_type.value}")
    return data_type


def setup_preprocessing_pipeline(config: Dict[str, Any], 
                               data_type: DataType, 
                               spark: Optional[SparkSession] = None) -> tuple:
    """Setup preprocessing components based on data type."""
    preprocessing = config.get('preprocessing', {})
    
    # Initialize sampler if specified
    sampler = None
    sampler_config = preprocessing.get('sampler')
    if sampler_config:
        sampler = factory.create_sampler(spark=spark, **sampler_config)
        logger.info("Configured data sampler")
    
    # Initialize normalizer for attribute-based data
    normalizer = None
    if data_type in [DataType.ATTRIBUTES, DataType.ATTRIBUTED_GRAPH]:
        normalizer_config = preprocessing.get('normalizer')
        if normalizer_config:
            normalizer = factory.create_normalizer(spark=spark, **normalizer_config)
            logger.info("Configured data normalizer")
    
    return sampler, normalizer


def create_data_loader(config: Dict[str, Any], 
                      data_type: DataType,
                      spark: Optional[SparkSession] = None,
                      sampler=None, 
                      normalizer=None):
    """Create appropriate data loader based on data type."""
    
    loader_config = {
        'spark': spark,
        'normalizer': normalizer,
        'sampler': sampler
    }
    
    if data_type == DataType.ATTRIBUTES:
        # Feature-only data
        loader_config.update({
            'features': config.get('features'),
            'similarity': None
        })
    elif data_type == DataType.GRAPH:
        # Graph-only data
        loader_config.update({
            'features': None,
            'similarity': config.get('similarity') or config.get('adjacency')
        })
    elif data_type == DataType.ATTRIBUTED_GRAPH:
        # Combined feature and graph data
        loader_config.update({
            'features': config.get('features'),
            'similarity': config.get('similarity') or config.get('adjacency')
        })
    
    return factory.create_loader(**loader_config)


def execute_clustering_pipeline(config: Dict[str, Any], 
                              data_loader, 
                              data_type: DataType) -> tuple:
    """Execute the clustering optimization pipeline."""
    
    # Validate algorithm compatibility with data type
    algorithm = config['algorithm']
    algorithm_info = MODEL_REGISTRY.get(algorithm)
    if not algorithm_info:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Check if algorithm supports the data type
    supported_types = algorithm_info.get('supported_data_types', [dt.value for dt in DataType])
    if data_type.value not in supported_types:
        logger.warning(f"Algorithm '{algorithm}' may not be optimized for data type '{data_type.value}'")
    
    # Initialize optimization components
    optimizer = factory.create_optimizer(config.get('optimizer', 'grid'))
    metric = factory.create_metric(config['metric'])
    model_class = algorithm_info['class']
    
    logger.info("Starting hyperparameter optimization...")
    best_params = optimizer.find_best(
        model_class=model_class,
        data_loader=data_loader,
        param_grid=config['params'],
        metric=metric
    )
    logger.info(f"Optimal parameters found: {best_params}")
    
    # Train final model with best parameters
    best_model = factory.create_model(algorithm, best_params)
    best_model.fit(data_loader)
    logger.info("Final model training completed")
    
    return best_model, best_params


def save_results(config: Dict[str, Any], 
                best_model, 
                data_loader, 
                data_type: DataType):
    """Save model, visualizations, and analysis results."""
    
    # Save trained model
    output_path = config.get('output_path')
    if output_path:
        best_model.save(output_path)
        logger.info(f"Model saved to: {output_path}")
    
    # Generate visualizations
    plots_path = config.get('plots_path')
    if plots_path:
        visualizer = factory.create_visualizer(plots_path)
        visualizer.visualisation(data_loader, best_model.labels_)
        logger.info(f"Visualizations saved to: {plots_path}")
    
    # Generate statistical analysis
    stat_path = config.get('stat_path')
    if stat_path:
        analyser = factory.create_analyser(stat_path)
        analyser.compute_statistics(data_loader, best_model.labels_)
        logger.info(f"Statistical analysis saved to: {stat_path}")


def print_help():
    """Display extended help information."""
    help_text = f"""
Pattern - Scalable Unsupervised Learning Library

SUPPORTED DATA TYPES:
  • Attributes/Features: Tabular data for feature-based clustering
  • Graph/Networks: Pure network data for graph clustering
  • Attributed Networks: Combined feature and graph data

PROCESSING MODES:
  • pandas: Single-machine processing
  • spark: Distributed processing with Apache Spark

AVAILABLE ALGORITHMS ({len(MODEL_REGISTRY)}):
{', '.join(MODEL_REGISTRY.keys())}

AVAILABLE METRICS ({len(METRIC_REGISTRY)}):
{', '.join(METRIC_REGISTRY.keys())}

USAGE EXAMPLES:
  1. Attribute-based clustering:
     python main.py config_attributes.json

  2. Graph clustering:
     python main.py config_graph.json

  3. Attributed network clustering:
     python main.py config_attributed_graph.json

  4. Algorithm-specific help:
     python main.py kmeans -h
"""
    print(help_text)


def handle_list_command():
    """Display detailed list of available algorithms and metrics."""
    print("=== IMPLEMENTED ALGORITHMS ===")
    for algo, info in MODEL_REGISTRY.items():
        params = ', '.join(info['params_help'].keys())
        supported_types = info.get('supported_data_types', ['all'])
        print(f"\n{algo.upper()}:")
        print(f"  Parameters: {params}")
        print(f"  Supported data types: {', '.join(supported_types)}")
    
    print("\n=== AVAILABLE METRICS ===")
    for metric_name in METRIC_REGISTRY.keys():
        print(f"  • {metric_name}")


@log_errors
def main():
    """Main application entry point."""
    # Initialize command line interface
    parser = create_root_parser()
    create_method_subparsers(parser)
    args = parser.parse_args()

    # Configure logging
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    # Handle help and listing commands
    if args.help:
        print_help()
        return
    
    if args.list:
        handle_list_command()
        return

    if not args.config_path:
        logger.error("Configuration file not specified")
        sys.exit(1)

    try:
        # Load and validate configuration
        config = load_config(args.config_path)
        logger.info(f"Configuration loaded from: {args.config_path}")
        
        # Determine processing mode and data type
        processing_mode = ProcessingMode(config.get('data_source', 'pandas'))
        data_type = validate_data_type_compatibility(config)
        
        # Execute pipeline with proper resource management
        with get_spark_session(processing_mode, config.get('spark_config')) as spark:
            
            # Setup preprocessing pipeline
            sampler, normalizer = setup_preprocessing_pipeline(config, data_type, spark)
            
            # Create data loader
            data_loader = create_data_loader(config, data_type, spark, sampler, normalizer)
            
            # Execute clustering pipeline
            best_model, best_params = execute_clustering_pipeline(config, data_loader, data_type)
            
            # Save results
            save_results(config, best_model, data_loader, data_type)
            
        logger.info("Pipeline execution completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        if args.debug:
            logger.exception("Full error traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()