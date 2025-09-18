"""
Configuration file for MovieLens Collaborative Filtering Recommendation System
Following academic standards and research practices
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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

class Config:
    """Main configuration class combining all sub-configurations"""

    def __init__(self):
        self.data = DataConfig()
        self.model = ModelConfig()
        self.evaluation = EvaluationConfig()
        self.visualization = VisualizationConfig()
        self.experiment = ExperimentConfig()

        # Create necessary directories
        self._create_directories()

    def _create_directories(self):
        """Create necessary directories for the experiment"""
        directories = [
            self.data.dataset_path,
            self.experiment.results_dir,
            self.experiment.logs_dir,
            self.experiment.plots_dir
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
            "experiment": self.experiment.__dict__
        }

# Global configuration instance
cfg = Config()