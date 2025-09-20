"""
Configuration file for MovieLens Collaborative Filtering Recommendation System
Following academic standards and research practices
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
from enum import Enum

class SearchMethod(Enum):
    """Hyperparameter search methods"""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"

class OptimizationObjective(Enum):
    """Optimization objectives for hyperparameter tuning"""
    MAE = "mae"
    RMSE = "rmse"
    PEARSON_CORRELATION = "pearson_correlation"
    PRECISION_AT_K = "precision_at_k"
    RECALL_AT_K = "recall_at_k"
    NDCG_AT_K = "ndcg_at_k"
    F1_AT_K = "f1_at_k"

@dataclass
class HyperparameterConfig:
    """Hyperparameter search configuration for academic experiments"""

    # Search method configuration
    search_method: SearchMethod = SearchMethod.GRID_SEARCH
    optimization_objective: OptimizationObjective = OptimizationObjective.RMSE
    maximize_objective: bool = False  # False for error metrics (MAE, RMSE), True for performance metrics

    # Cross-validation settings
    cv_folds: int = 5
    cv_random_state: int = 42

    # Search space configuration
    # User-based CF parameters
    user_k_neighbors_range: List[int] = None
    user_similarity_metrics: List[str] = None

    # Item-based CF parameters
    item_k_neighbors_range: List[int] = None
    item_similarity_metrics: List[str] = None

    # Data preprocessing parameters
    min_ratings_per_user_range: List[int] = None
    min_ratings_per_item_range: List[int] = None
    train_ratio_range: List[float] = None

    # General model parameters
    prediction_threshold_range: List[float] = None

    # Search algorithm parameters
    n_iter_random_search: int = 100  # For random search
    n_initial_points: int = 10       # For Bayesian optimization
    acquisition_function: str = "EI" # Expected Improvement for Bayesian opt

    # Parallel processing
    n_jobs: int = -1  # Use all available cores

    # Statistical testing
    perform_statistical_tests: bool = True
    statistical_test_method: str = "wilcoxon"  # wilcoxon, t_test, friedman
    significance_level: float = 0.05

    # Early stopping
    enable_early_stopping: bool = False
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001

    # Multi-objective configuration
    secondary_objectives: List[OptimizationObjective] = None
    objective_weights: Dict[str, float] = None
    multi_objective_strategy: str = "weighted_sum"  # weighted_sum, pareto
    objective_directions: Dict[str, bool] = field(default_factory=dict)

    # Search management
    enable_progress_bar: bool = True
    progress_update_interval: int = 5
    intermediate_save_frequency: int = 5
    prune_worse_than: Optional[float] = None
    prune_relative: bool = True

    # Parallel execution
    use_parallel_evaluation: bool = True
    parallel_backend: str = "thread"

    def __post_init__(self):
        if self.user_k_neighbors_range is None:
            self.user_k_neighbors_range = [10, 20, 30, 50, 75, 100]

        if self.user_similarity_metrics is None:
            self.user_similarity_metrics = ["cosine", "pearson"]

        if self.item_k_neighbors_range is None:
            self.item_k_neighbors_range = [10, 20, 30, 50, 75, 100]

        if self.item_similarity_metrics is None:
            self.item_similarity_metrics = ["cosine", "pearson"]

        if self.min_ratings_per_user_range is None:
            self.min_ratings_per_user_range = [5, 10, 15, 20]

        if self.min_ratings_per_item_range is None:
            self.min_ratings_per_item_range = [5, 10, 15, 20]

        if self.train_ratio_range is None:
            self.train_ratio_range = [0.7, 0.8, 0.9]

        if self.prediction_threshold_range is None:
            self.prediction_threshold_range = [2.5, 3.0, 3.5, 4.0]

        if self.secondary_objectives is None:
            self.secondary_objectives = []

        # Normalize objective weights and directions
        maximize_objectives = {
            OptimizationObjective.PEARSON_CORRELATION,
            OptimizationObjective.PRECISION_AT_K,
            OptimizationObjective.RECALL_AT_K,
            OptimizationObjective.NDCG_AT_K,
            OptimizationObjective.F1_AT_K
        }

        all_objectives = [self.optimization_objective] + list(self.secondary_objectives)
        if not all_objectives:
            all_objectives = [OptimizationObjective.RMSE]

        # Build direction map (True -> maximize, False -> minimize)
        self.objective_directions = {
            (obj.value if isinstance(obj, OptimizationObjective) else str(obj)): (obj in maximize_objectives)
            for obj in all_objectives
        }

        # Initialize objective weights if not provided
        if self.objective_weights is None:
            self.objective_weights = {key: 1.0 for key in self.objective_directions.keys()}
        else:
            normalized_weights = {}
            for key, weight in self.objective_weights.items():
                normalized_weights[str(key)] = float(weight)
            for key in self.objective_directions.keys():
                normalized_weights.setdefault(key, 1.0)
            self.objective_weights = normalized_weights

        # Primary objective direction controls legacy flag
        primary_key = self.optimization_objective.value
        self.maximize_objective = self.objective_directions.get(primary_key, False)

        if self.multi_objective_strategy not in {"weighted_sum", "pareto"}:
            raise ValueError(f"Unsupported multi-objective strategy: {self.multi_objective_strategy}")

        if self.progress_update_interval < 1:
            self.progress_update_interval = 1

        if self.intermediate_save_frequency < 1:
            self.intermediate_save_frequency = 1

        if self.prune_worse_than is not None and self.prune_worse_than < 0:
            raise ValueError("prune_worse_than must be >= 0")

@dataclass
class DataConfig:
    """Data configuration parameters"""
    dataset_path: str = "./data"
    ratings_file: str = "ratings.csv"
    movies_file: str = "movies.csv"
    train_ratio: float = 0.8
    test_ratio: float = 0.2
    min_ratings_per_user: int = 5
    min_ratings_per_item: int = 5

@dataclass
class ModelConfig:
    """Model configuration parameters"""
    # User-based CF parameters
    user_similarity_metric: str = "cosine"  # cosine, pearson, jaccard
    user_k_neighbors: int = 50

    # Item-based CF parameters
    item_similarity_metric: str = "cosine"  # cosine, pearson, jaccard
    item_k_neighbors: int = 50

    # General parameters
    rating_scale: Tuple[float, float] = (0.5, 5.0)
    prediction_threshold: float = 3.0
    backend: str = "numpy"           # numpy, torch
    device: str = "cpu"              # cpu, cuda, cuda:0, etc.

@dataclass
class EvaluationConfig:
    """Evaluation configuration parameters"""
    metrics: List[str] = None
    top_k_recommendations: List[int] = None
    cross_validation_folds: int = 5

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ["mae", "rmse", "precision", "recall", "f1", "ndcg"]
        if self.top_k_recommendations is None:
            self.top_k_recommendations = [5, 10, 20, 50]

@dataclass
class VisualizationConfig:
    """Visualization configuration parameters"""
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 300
    style: str = "seaborn-v0_8"
    color_palette: str = "viridis"
    save_plots: bool = True
    plot_formats: List[str] = None

    def __post_init__(self):
        if self.plot_formats is None:
            self.plot_formats = ["png", "pdf"]

@dataclass
class ExperimentConfig:
    """Experiment configuration following academic research standards"""
    experiment_name: str = "movielens_cf_baseline"
    base_results_dir: str = "./results"
    base_logs_dir: str = "./logs"
    base_plots_dir: str = "./plots"
    results_dir: str = "./results"
    logs_dir: str = "./logs"
    plots_dir: str = "./plots"
    random_seed: int = 42
    verbose: bool = True
    save_intermediate_results: bool = True

    # Academic reporting parameters
    report_format: str = "academic"  # academic, technical, brief
    include_statistical_tests: bool = True
    significance_level: float = 0.05

    # Runtime metadata
    current_run_id: Optional[str] = None
    current_run_timestamp: Optional[str] = None
    current_run_mode: Optional[str] = None
    current_run_root: Optional[str] = None

    def reset_to_base_dirs(self):
        self.results_dir = self.base_results_dir
        self.logs_dir = self.base_logs_dir
        self.plots_dir = self.base_plots_dir

class Config:
    """Main configuration class combining all sub-configurations"""

    def __init__(self):
        self.data = DataConfig()
        self.model = ModelConfig()
        self.evaluation = EvaluationConfig()
        self.visualization = VisualizationConfig()
        self.experiment = ExperimentConfig()
        self.experiment.reset_to_base_dirs()
        self.hyperparameter = HyperparameterConfig()

        # Create necessary directories
        self._create_directories()

    def _create_directories(self):
        """Create necessary directories for the experiment"""
        directories = [
            self.data.dataset_path,
            self.experiment.base_results_dir,
            self.experiment.base_logs_dir,
            self.experiment.base_plots_dir
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def update_config(self, **kwargs):
        """Update configuration parameters"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")

    def get_experiment_path(self, filename: str) -> str:
        """Get full path for experiment files"""
        return os.path.join(self.experiment.results_dir, filename)

    def get_plot_path(self, filename: str) -> str:
        """Get full path for plot files"""
        return os.path.join(self.experiment.plots_dir, filename)

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary for logging"""
        return {
            "data": self.data.__dict__,
            "model": self.model.__dict__,
            "evaluation": self.evaluation.__dict__,
            "visualization": self.visualization.__dict__,
            "experiment": self.experiment.__dict__,
            "hyperparameter": self.hyperparameter.__dict__
        }

# Global configuration instance
cfg = Config()
