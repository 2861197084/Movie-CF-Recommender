"""
Main execution script for MovieLens Collaborative Filtering Experiments
Following academic research standards and best practices
"""

import os
import sys
import argparse
import json
import time
import re
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Local imports
from config import cfg
from utils.logger import get_logger
from utils.data_loader import load_movielens_data
from models.user_based_cf import UserBasedCollaborativeFiltering
from models.item_based_cf import ItemBasedCollaborativeFiltering
from models.cf_factory import build_cf_model
from evaluation.metrics import MetricsEvaluator
from analysis.visualizer import AcademicVisualizer
from analysis.hyperparameter_optimizer import AcademicHyperparameterOptimizer
from analysis.academic_reporter import AcademicReporter

def _slugify_token(value: Optional[str]) -> str:
    if not value:
        return 'default'
    token = re.sub(r'[^a-zA-Z0-9]+', '-', value.strip().lower())
    token = re.sub(r'-+', '-', token).strip('-')
    return token or 'default'

def prepare_run_environment(mode: str, dataset: str, extras: Optional[Dict[str, str]] = None) -> str:
    """Create run-specific output directories and update experiment config"""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    extras = extras or {}

    tokens = [timestamp, mode, dataset, cfg.experiment.experiment_name]
    for key, value in extras.items():
        tokens.append(f"{key}-{value}")

    run_id = '_'.join(_slugify_token(token) for token in tokens if token)

    results_dir = os.path.join(cfg.experiment.base_results_dir, run_id)
    plots_dir = os.path.join(cfg.experiment.base_plots_dir, run_id)
    logs_dir = os.path.join(cfg.experiment.base_logs_dir, run_id)

    for directory in (results_dir, plots_dir, logs_dir):
        os.makedirs(directory, exist_ok=True)

    cfg.experiment.current_run_id = run_id
    cfg.experiment.current_run_timestamp = timestamp
    cfg.experiment.current_run_mode = mode
    cfg.experiment.current_run_root = results_dir
    cfg.experiment.results_dir = results_dir
    cfg.experiment.plots_dir = plots_dir
    cfg.experiment.logs_dir = logs_dir

    return run_id


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

    # Initialize models based on selected backend
    model_specs = [
        ("UserCF_Cosine", 'user_cf', {
            'similarity_metric': 'cosine',
            'k_neighbors': cfg.model.user_k_neighbors
        }),
        ("UserCF_Pearson", 'user_cf', {
            'similarity_metric': 'pearson',
            'k_neighbors': cfg.model.user_k_neighbors
        }),
        ("ItemCF_Cosine", 'item_cf', {
            'similarity_metric': 'cosine',
            'k_neighbors': cfg.model.item_k_neighbors
        }),
        ("ItemCF_Pearson", 'item_cf', {
            'similarity_metric': 'pearson',
            'k_neighbors': cfg.model.item_k_neighbors
        })
    ]

    models = {}
    for name, mtype, params in model_specs:
        try:
            model = build_cf_model(
                mtype,
                params,
                backend=cfg.model.backend,
                device=cfg.model.device
            )
        except Exception as exc:
            logger.warning(f"Skipping {name} due to backend limitation: {exc}")
            continue
        models[name] = model

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

def run_hyperparameter_search(logger, dataset_name: str = "ml-latest-small") -> Dict:
    """
    Run hyperparameter optimization experiments

    Args:
        logger: Logger instance
        dataset_name: Name of MovieLens dataset to use

    Returns:
        Dictionary containing optimization results
    """
    logger.log_experiment_start(cfg.to_dict())
    logger.log_phase("Starting Hyperparameter Optimization")

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

    # Initialize hyperparameter optimizer
    optimizer = AcademicHyperparameterOptimizer(cfg.hyperparameter)

    try:
        # Run optimization
        logger.log_phase("Running Hyperparameter Optimization")
        hp_result = optimizer.optimize(loader, user_item_matrix, test_df)

        # Save optimization results
        logger.log_phase("Saving Hyperparameter Search Results")
        results_file = optimizer.save_results(
            hp_result, os.path.join(cfg.experiment.results_dir, "hyperparameter_results.json")
        )

        # Generate hyperparameter analysis visualizations
        logger.log_phase("Generating Hyperparameter Analysis Visualizations")
        visualizer = AcademicVisualizer(cfg)

        try:
            # Main hyperparameter analysis plot
            hp_fig = visualizer.plot_hyperparameter_analysis(hp_result, cfg.experiment.plots_dir)
            logger.info("Generated hyperparameter analysis plot")

            # Create interactive dashboard
            dashboard_path = visualizer.create_hyperparameter_dashboard(
                hp_result, cfg.experiment.plots_dir
            )
            if dashboard_path:
                logger.info(f"Generated interactive dashboard: {dashboard_path}")

            # Generate specific parameter heatmaps for key parameters
            key_params = ['k_neighbors', 'min_ratings_per_user', 'train_ratio']
            for i, param1 in enumerate(key_params[:-1]):
                for param2 in key_params[i+1:]:
                    try:
                        heatmap_fig = visualizer.plot_hyperparameter_heatmap(
                            hp_result, param1, param2, cfg.experiment.plots_dir
                        )
                        logger.info(f"Generated heatmap: {param1} vs {param2}")
                        plt.close(heatmap_fig)
                    except Exception as e:
                        logger.warning(f"Failed to generate heatmap {param1} vs {param2}: {e}")

            plt.close(hp_fig)

        except Exception as e:
            logger.error(f"Error generating hyperparameter visualizations: {e}")

        # Train best model with full dataset for comparison
        logger.log_phase("Training Best Model on Full Dataset")
        best_params = hp_result.best_params

        best_model = build_cf_model(
            best_params['model_type'],
            {
                'similarity_metric': best_params['similarity_metric'],
                'k_neighbors': best_params['k_neighbors']
            },
            backend=cfg.model.backend,
            device=cfg.model.device
        )

        # Train on full dataset
        best_model.fit(user_item_matrix)

        # Evaluate best model
        evaluator = MetricsEvaluator()
        best_model_results = evaluator.comprehensive_evaluation(
            model=best_model,
            test_data=test_df,
            user_item_matrix=user_item_matrix,
            k_values=cfg.evaluation.top_k_recommendations,
            threshold=cfg.model.prediction_threshold,
            data_loader=loader
        )

        logger.log_experiment_end()

        return {
            'hyperparameter_results': hp_result,
            'best_model_evaluation': best_model_results,
            'dataset_stats': stats,
            'data_loader': loader,
            'results_file': results_file
        }

    except Exception as e:
        logger.error(f"Hyperparameter optimization failed: {e}")
        raise

def print_hyperparameter_summary(results: Dict):
    """
    Print a summary of hyperparameter optimization results

    Args:
        results: Hyperparameter optimization results dictionary
    """
    print("\n" + "="*80)
    print("HYPERPARAMETER OPTIMIZATION RESULTS")
    print("="*80)

    hp_result = results['hyperparameter_results']
    stats = results['dataset_stats']

    # Dataset summary
    print(f"\nDataset Statistics:")
    print(f"  Users: {stats['n_users']:,}")
    print(f"  Items: {stats['n_items']:,}")
    print(f"  Ratings: {stats['n_ratings']:,}")
    print(f"  Sparsity: {stats['sparsity']:.4f}")

    # Optimization summary
    print(f"\nOptimization Summary:")
    print(f"  Search Method: {cfg.hyperparameter.search_method.value}")
    print(f"  Optimization Objective: {cfg.hyperparameter.optimization_objective.value}")
    print(f"  Total Trials: {len(hp_result.all_results)}")
    print(f"  Best Score: {hp_result.best_score:.6f}")
    print(f"  Execution Time: {hp_result.execution_time:.2f} seconds")

    # Best configuration
    print(f"\nBest Configuration:")
    print(f"  Model Type: {hp_result.best_model_type.replace('_', ' ').upper()}")
    if cfg.hyperparameter.objective_weights:
        print("\nObjective Weights:")
        for name, weight in cfg.hyperparameter.objective_weights.items():
            label = name.replace('_', ' ').title()
            print(f"  {label}: {weight:.2f}")

    if hp_result.best_objective_scores:
        print("\nBest Objective Scores:")
        for name, score in hp_result.best_objective_scores.items():
            label = name.replace('_', ' ').title()
            print(f"  {label}: {score:.4f}")

    if hp_result.objective_summary:
        mean_summary = hp_result.objective_summary.get('mean', {})
        if mean_summary:
            print("Objective Score Averages (Across Trials):")
            for name, score in mean_summary.items():
                label = name.replace('_', ' ').title()
                print(f"  {label}: {score:.4f}")

    for param, value in hp_result.best_params.items():
        print(f"  {param.replace('_', ' ').title()}: {value}")

    # Statistical analysis
    if hp_result.statistical_analysis:
        stats_analysis = hp_result.statistical_analysis
        print(f"\nStatistical Analysis:")
        print(f"  Overall Mean Score: {stats_analysis.get('overall_mean', 0):.6f}")
        print(f"  Overall Std Score: {stats_analysis.get('overall_std', 0):.6f}")

        if 'statistical_test' in stats_analysis:
            test_info = stats_analysis['statistical_test']
            print(f"  Statistical Test: {test_info['method'].title()}")
            print(f"  P-value: {test_info.get('p_value', 0):.4f}")
            significance = "Yes" if test_info.get('significant', False) else "No"
            print(f"  Statistically Significant: {significance}")

    # Convergence info
    if hp_result.convergence_info:
        conv_info = hp_result.convergence_info
        print(f"\nConvergence Information:")
        print(f"  Best Found at Iteration: {conv_info.get('best_iteration', 'N/A')}")
        print(f"  Early Stopping: {'Yes' if conv_info.get('early_stopping_triggered', False) else 'No'}")

    print("\n" + "="*80)

def save_results(results: Dict, stats: Dict, results_dir: str, logger):
    """
    Save experimental results in multiple formats

    Args:
        results: Model evaluation results
        stats: Dataset statistics
        results_dir: Directory to save results
        logger: Logger instance
    """
    run_timestamp = cfg.experiment.current_run_timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")

    # Save comprehensive results as JSON
    results_file = os.path.join(results_dir, "results.json")
    comprehensive_results = {
        'experiment_config': cfg.to_dict(),
        'dataset_statistics': stats,
        'model_results': results,
        'run_metadata': {
            'run_id': cfg.experiment.current_run_id,
            'timestamp': run_timestamp,
            'mode': cfg.experiment.current_run_mode,
            'experiment_name': cfg.experiment.experiment_name
        }
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
            summary_file = os.path.join(results_dir, "summary.csv")
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
    parser.add_argument('--hyperparameter-search', action='store_true',
                       help='Run hyperparameter optimization instead of baseline experiments')
    parser.add_argument('--search-method', type=str, default='grid_search',
                       choices=['grid_search', 'random_search'],
                       help='Hyperparameter search method')
    parser.add_argument('--n-trials', type=int, default=50,
                       help='Number of trials for random search')
    parser.add_argument('--cv-folds', type=int, default=3,
                       help='Number of cross-validation folds for hyperparameter search')
    parser.add_argument('--generate-report-only', action='store_true',
                       help='Generate academic report from existing results (skip experiments)')
    parser.add_argument('--backend', type=str, default=None,
                        choices=['numpy', 'torch'],
                        help='Backend used for collaborative filtering models')
    parser.add_argument('--device', type=str, default=None,
                        help='Device identifier for torch backend (e.g. cuda, cuda:0, cpu)')

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

    # Configure hyperparameter search
    if args.hyperparameter_search:
        from config import SearchMethod
        cfg.hyperparameter.search_method = SearchMethod.GRID_SEARCH if args.search_method == 'grid_search' else SearchMethod.RANDOM_SEARCH
        cfg.hyperparameter.n_iter_random_search = args.n_trials
        cfg.hyperparameter.cv_folds = args.cv_folds

        # For quick test with hyperparameter search, reduce search space
        if args.quick_test:
            cfg.hyperparameter.user_k_neighbors_range = [10, 20, 30]
            cfg.hyperparameter.item_k_neighbors_range = [10, 20, 30]
            cfg.hyperparameter.min_ratings_per_user_range = [5, 10]
            cfg.hyperparameter.min_ratings_per_item_range = [5, 10]
            cfg.hyperparameter.train_ratio_range = [0.8]
            cfg.hyperparameter.prediction_threshold_range = [3.0]
            cfg.hyperparameter.n_iter_random_search = 20

    if args.experiment_name:
        cfg.experiment.experiment_name = args.experiment_name

    if args.backend:
        cfg.model.backend = args.backend
    if args.device:
        cfg.model.device = args.device

    run_mode = 'hyperparameter' if args.hyperparameter_search else 'baseline'
    extras: Dict[str, str] = {}
    if args.hyperparameter_search:
        extras['search'] = args.search_method
    if args.quick_test:
        extras['quick'] = 'true'

    prepare_run_environment(run_mode, args.dataset, extras)

    # Setup experiment
    logger = setup_experiment(None)
    logger.info(f"Run directory: {cfg.experiment.current_run_root}")

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
        if args.hyperparameter_search:
            # Run hyperparameter optimization
            results = run_hyperparameter_search(logger, args.dataset)
            # Print hyperparameter summary
            print_hyperparameter_summary(results)

            # Generate academic report
            logger.info("Generating academic report for hyperparameter optimization")
            reporter = AcademicReporter(cfg)
            try:
                report_path = reporter.generate_full_report(hp_results=results)
                if report_path:
                    print(f"\nAcademic report generated: {report_path}")
                    print(f"Markdown version: {report_path.replace('.tex', '.md')}")
            except Exception as e:
                logger.warning(f"Failed to generate academic report: {e}")

        else:
            # Run baseline experiments
            results = run_baseline_experiments(logger, args.dataset)
            # Print baseline summary
            print_results_summary(results)

            # Generate academic report
            logger.info("Generating academic report for baseline experiments")
            reporter = AcademicReporter(cfg)
            try:
                report_path = reporter.generate_full_report(baseline_results=results)
                if report_path:
                    print(f"\nAcademic report generated: {report_path}")
                    print(f"Markdown version: {report_path.replace('.tex', '.md')}")
            except Exception as e:
                logger.warning(f"Failed to generate academic report: {e}")

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
