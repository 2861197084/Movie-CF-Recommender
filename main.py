"""
Main execution script for MovieLens Collaborative Filtering Experiments
Following academic research standards and best practices
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Local imports
from config import cfg
from utils.logger import get_logger
from utils.data_loader import load_movielens_data
from models.user_based_cf import UserBasedCollaborativeFiltering
from models.item_based_cf import ItemBasedCollaborativeFiltering
from evaluation.metrics import MetricsEvaluator
from analysis.visualizer import AcademicVisualizer

def setup_experiment(experiment_name: Optional[str] = None):
    """
    Setup experiment environment and logging

    Args:
        experiment_name: Custom experiment name
    """
    if experiment_name:
        cfg.experiment.experiment_name = experiment_name

    # Set random seeds for reproducibility
    np.random.seed(cfg.experiment.random_seed)

    # Create logger
    logger = get_logger("MainExperiment", cfg.experiment.logs_dir)

    return logger

def run_baseline_experiments(logger, dataset_name: str = "ml-latest-small") -> Dict:
    """
    Run baseline collaborative filtering experiments

    Args:
        logger: Logger instance
        dataset_name: Name of MovieLens dataset to use

    Returns:
        Dictionary containing all experimental results
    """
    logger.log_experiment_start(cfg.to_dict())

    # Load and preprocess data
    logger.log_phase("Data Loading and Preprocessing")
    loader, train_df, test_df, user_item_matrix = load_movielens_data(
        config=cfg,
        dataset_name=dataset_name,
        auto_download=True
    )

    # Log dataset statistics
    stats = loader.get_dataset_statistics()
    logger.log_metrics(stats, "Dataset Statistics")

    # Initialize models
    models = {
        "UserCF_Cosine": UserBasedCollaborativeFiltering(
            similarity_metric="cosine",
            k_neighbors=cfg.model.user_k_neighbors
        ),
        "UserCF_Pearson": UserBasedCollaborativeFiltering(
            similarity_metric="pearson",
            k_neighbors=cfg.model.user_k_neighbors
        ),
        "ItemCF_Cosine": ItemBasedCollaborativeFiltering(
            similarity_metric="cosine",
            k_neighbors=cfg.model.item_k_neighbors
        ),
        "ItemCF_Pearson": ItemBasedCollaborativeFiltering(
            similarity_metric="pearson",
            k_neighbors=cfg.model.item_k_neighbors
        )
    }

    # Train and evaluate models
    results = {}
    evaluator = MetricsEvaluator()

    for model_name, model in models.items():
        logger.log_phase(f"Training and Evaluating {model_name}")

        try:
            # Train model
            start_time = time.time()
            model.fit(user_item_matrix)
            training_time = time.time() - start_time

            logger.info(f"Model {model_name} trained in {training_time:.2f} seconds")

            # Evaluate model
            start_time = time.time()
            model_results = evaluator.comprehensive_evaluation(
                model=model,
                test_data=test_df,
                user_item_matrix=user_item_matrix,
                k_values=cfg.evaluation.top_k_recommendations,
                threshold=cfg.model.prediction_threshold,
                data_loader=loader
            )
            evaluation_time = time.time() - start_time

            # Add timing information
            model_results['timing'] = {
                'training_time': training_time,
                'evaluation_time': evaluation_time
            }

            results[model_name] = model_results

            # Log key metrics
            if 'rating_prediction' in model_results:
                key_metrics = {
                    'MAE': model_results['rating_prediction'].get('mae', 0),
                    'RMSE': model_results['rating_prediction'].get('rmse', 0),
                    'Correlation': model_results['rating_prediction'].get('pearson_correlation', 0)
                }
                logger.log_metrics(key_metrics, f"{model_name} Performance")

        except Exception as e:
            logger.error(f"Error training/evaluating {model_name}: {e}")
            results[model_name] = {'error': str(e)}

    # Generate visualizations
    logger.log_phase("Generating Visualizations")
    visualizer = AcademicVisualizer(cfg)

    try:
        plot_paths = visualizer.generate_all_visualizations(
            loader=loader,
            models_results=results,
            save_path=cfg.experiment.plots_dir
        )
        logger.info(f"Generated visualizations: {list(plot_paths.keys())}")

    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")

    # Save results
    logger.log_phase("Saving Results")
    save_results(results, stats, cfg.experiment.results_dir, logger)

    logger.log_experiment_end()

    return {
        'models_results': results,
        'dataset_stats': stats,
        'data_loader': loader
    }

def save_results(results: Dict, stats: Dict, results_dir: str, logger):
    """
    Save experimental results in multiple formats

    Args:
        results: Model evaluation results
        stats: Dataset statistics
        results_dir: Directory to save results
        logger: Logger instance
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save comprehensive results as JSON
    results_file = os.path.join(results_dir, f"results_{timestamp}.json")
    comprehensive_results = {
        'experiment_config': cfg.to_dict(),
        'dataset_statistics': stats,
        'model_results': results,
        'timestamp': timestamp
    }

    try:
        with open(results_file, 'w') as f:
            json.dump(comprehensive_results, f, indent=2, default=str)
        logger.info(f"Results saved to: {results_file}")

    except Exception as e:
        logger.error(f"Error saving results: {e}")

    # Save summary CSV
    try:
        summary_data = []
        for model_name, model_results in results.items():
            if 'error' not in model_results and 'rating_prediction' in model_results:
                rating_metrics = model_results['rating_prediction']
                summary_data.append({
                    'Model': model_name,
                    'MAE': rating_metrics.get('mae', 0),
                    'RMSE': rating_metrics.get('rmse', 0),
                    'Pearson_Correlation': rating_metrics.get('pearson_correlation', 0),
                    'Training_Time': model_results.get('timing', {}).get('training_time', 0),
                    'Evaluation_Time': model_results.get('timing', {}).get('evaluation_time', 0)
                })

        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_file = os.path.join(results_dir, f"summary_{timestamp}.csv")
            summary_df.to_csv(summary_file, index=False)
            logger.info(f"Summary saved to: {summary_file}")

    except Exception as e:
        logger.error(f"Error saving summary: {e}")

def print_results_summary(results: Dict):
    """
    Print a concise summary of experimental results

    Args:
        results: Experimental results dictionary
    """
    print("\n" + "="*80)
    print("MOVIELENS COLLABORATIVE FILTERING - EXPERIMENTAL RESULTS")
    print("="*80)

    models_results = results['models_results']
    stats = results['dataset_stats']

    # Dataset summary
    print(f"\nDataset Statistics:")
    print(f"  Users: {stats['n_users']:,}")
    print(f"  Items: {stats['n_items']:,}")
    print(f"  Ratings: {stats['n_ratings']:,}")
    print(f"  Sparsity: {stats['sparsity']:.4f}")

    # Model performance summary
    print(f"\nModel Performance Summary:")
    print(f"{'Model':<20} {'MAE':<8} {'RMSE':<8} {'Correlation':<12} {'Time(s)':<10}")
    print("-" * 70)

    for model_name, model_results in models_results.items():
        if 'error' in model_results:
            print(f"{model_name:<20} {'ERROR':<8} {'ERROR':<8} {'ERROR':<12} {'ERROR':<10}")
        elif 'rating_prediction' in model_results:
            rating_metrics = model_results['rating_prediction']
            timing = model_results.get('timing', {})

            mae = rating_metrics.get('mae', 0)
            rmse = rating_metrics.get('rmse', 0)
            corr = rating_metrics.get('pearson_correlation', 0)
            time_taken = timing.get('training_time', 0) + timing.get('evaluation_time', 0)

            print(f"{model_name:<20} {mae:<8.4f} {rmse:<8.4f} {corr:<12.4f} {time_taken:<10.2f}")

    print("\n" + "="*80)

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="MovieLens Collaborative Filtering Experiments")
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Custom experiment name')
    parser.add_argument('--data-path', type=str, default=None,
                       help='Path to MovieLens dataset')
    parser.add_argument('--dataset', type=str, default="ml-latest-small",
                       choices=['ml-latest-small', 'ml-latest', 'ml-25m', 'ml-100k'],
                       help='MovieLens dataset to use (will auto-download if not found)')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to custom configuration file')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test with smaller dataset/parameters')
    parser.add_argument('--no-download', action='store_true',
                       help='Disable automatic dataset download')

    args = parser.parse_args()

    # Update configuration if provided
    if args.data_path:
        cfg.data.dataset_path = args.data_path

    if args.quick_test:
        cfg.data.min_ratings_per_user = 10
        cfg.data.min_ratings_per_item = 10
        cfg.model.user_k_neighbors = 20
        cfg.model.item_k_neighbors = 20
        cfg.evaluation.top_k_recommendations = [5, 10]
        # Use smallest dataset for quick test
        args.dataset = "ml-latest-small"

    # Setup experiment
    logger = setup_experiment(args.experiment_name)

    # Print dataset information
    if not args.no_download:
        from utils.dataset_downloader import MovieLensDownloader
        downloader = MovieLensDownloader()
        available_datasets = downloader.list_available_datasets()

        logger.info("Available MovieLens datasets:")
        for dataset in available_datasets:
            status = "✓ Selected" if dataset["name"] == args.dataset else "  Available"
            logger.info(f"  {status}: {dataset['name']} - {dataset['description']}")

    try:
        # Run experiments
        results = run_baseline_experiments(logger, args.dataset)

        # Print summary
        print_results_summary(results)

        # Print file locations
        print(f"\nResults saved in: {cfg.experiment.results_dir}")
        print(f"Plots saved in: {cfg.experiment.plots_dir}")
        print(f"Logs saved in: {cfg.experiment.logs_dir}")

    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        if "dataset" in str(e).lower() or "download" in str(e).lower():
            logger.info("\nTroubleshooting:")
            logger.info("1. Check your internet connection")
            logger.info("2. Try a different dataset: --dataset ml-100k")
            logger.info("3. Manually download dataset to ./data/ directory")
            logger.info("4. Use --no-download flag and provide local dataset")
        sys.exit(1)

if __name__ == "__main__":
    main()