"""
Hyperparameter Optimization Module for Collaborative Filtering
Following academic research standards and best practices
"""

import os
import json
import time
import itertools
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Iterable, Callable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.model_selection import KFold

from config import cfg, SearchMethod, OptimizationObjective, HyperparameterConfig
from utils.logger import get_logger
from utils.data_loader import MovieLensLoader
from models.user_based_cf import UserBasedCollaborativeFiltering
from models.item_based_cf import ItemBasedCollaborativeFiltering
from models.cf_factory import build_cf_model
from evaluation.metrics import MetricsEvaluator, RecommenderMetrics


@dataclass
class HyperparameterSearchResult:
    """Results from hyperparameter search"""
    best_params: Dict[str, Any]
    best_score: float
    best_model_type: str
    all_results: List[Dict[str, Any]]
    search_history: Dict[str, Any]
    statistical_analysis: Optional[Dict[str, Any]] = None
    execution_time: float = 0.0
    convergence_info: Optional[Dict[str, Any]] = None
    best_objective_scores: Optional[Dict[str, float]] = None
    objective_summary: Optional[Dict[str, Dict[str, float]]] = None


class AcademicHyperparameterOptimizer:
    """
    Academic-standard hyperparameter optimizer for collaborative filtering models

    Implements rigorous hyperparameter search methodologies that include:
    - Grid Search and Random Search algorithms
    - k-fold Cross-validation with optional parallel execution
    - Statistical significance testing
    - Early stopping with configurable tolerance
    - Search space pruning heuristics
    - Multi-objective optimisation with weighted aggregation
    - Intermediate result persistence for long experiments
    """

    def __init__(self, config: HyperparameterConfig = None, logger_name: str = "HyperparameterOptimizer"):
        self.config = config or cfg.hyperparameter
        self.logger = get_logger(logger_name)
        self.evaluator = MetricsEvaluator()
        self.metrics_backend = RecommenderMetrics()
        self.model_backend = cfg.model.backend
        self.model_device = cfg.model.device

        self.objectives = self._resolve_objectives()
        self.objective_keys = [obj.value for obj in self.objectives]
        self.objective_weights = {
            key: float(self.config.objective_weights.get(key, 1.0))
            for key in self.objective_keys
        }

        self.k_values = list(cfg.evaluation.top_k_recommendations)
        self.primary_k = self.k_values[0] if self.k_values else 10
        self.relevance_threshold = cfg.model.prediction_threshold

        self.search_history: List[Dict[str, Any]] = []
        self.pruned_configurations = set()

        # Aggregated objective is always minimised internally
        self.minimize_score = True
        self.best_score = float("inf")
        self.best_params = None
        self.best_model_type = None
        self.best_iteration = 0
        self.best_objective_scores = None
        self.patience_counter = 0

        self.completed_evaluations = 0
        self.last_progress_update = time.time()
        self.intermediate_results_path = os.path.join(
            cfg.experiment.results_dir, "hyperparameter_search_intermediate.json"
        )
        self._cached_cv_splits: Optional[List[Tuple[pd.DataFrame, pd.DataFrame]]] = None

        self.parallel_backend = self.config.parallel_backend.lower()
        if self.parallel_backend not in {"thread", "process"}:
            self.logger.warning(
                f"Unknown parallel backend '{self.parallel_backend}', falling back to thread pool."
            )
            self.parallel_backend = "thread"

        self._prepare_encoding_metadata()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def optimize(self, data_loader: MovieLensLoader, user_item_matrix,
                 test_data: pd.DataFrame, timestamp_matrix=None) -> HyperparameterSearchResult:
        """Run the configured hyperparameter optimisation routine"""
        self.logger.info(
            f"Starting hyperparameter optimisation with {self.config.search_method.value}"
        )
        self.logger.info(
            "Primary objective: %s | Secondary objectives: %s" % (
                self.config.optimization_objective.value,
                [obj.value for obj in self.config.secondary_objectives]
            )
        )

        start_time = time.time()
        self._cached_cv_splits = None

        if self.config.search_method == SearchMethod.GRID_SEARCH:
            param_combinations = self._generate_grid_search_combinations()
            self.logger.info(f"Total parameter configurations to evaluate: {len(param_combinations)}")
            results = self._evaluate_parameter_combinations(
                param_combinations, data_loader, user_item_matrix, timestamp_matrix
            )
        elif self.config.search_method == SearchMethod.RANDOM_SEARCH:
            param_combinations = self._generate_random_search_combinations()
            self.logger.info(f"Random search iterations: {len(param_combinations)}")
            results = self._evaluate_parameter_combinations(
                param_combinations, data_loader, user_item_matrix, timestamp_matrix
            )
        elif self.config.search_method == SearchMethod.BAYESIAN_OPTIMIZATION:
            results = self._run_bayesian_optimization(data_loader, user_item_matrix, timestamp_matrix)
        else:
            raise NotImplementedError(
                f"Search method {self.config.search_method} is not implemented."
            )

        execution_time = time.time() - start_time
        statistical_analysis = None
        if self.config.perform_statistical_tests and len(results) > 1:
            statistical_analysis = self._perform_statistical_analysis(results)

        search_result = HyperparameterSearchResult(
            best_params=self.best_params or {},
            best_score=float(self.best_score),
            best_model_type=self.best_model_type or "unknown",
            all_results=results,
            search_history=self.search_history,
            statistical_analysis=statistical_analysis,
            execution_time=execution_time,
            convergence_info=self._get_convergence_info(),
            best_objective_scores=self.best_objective_scores,
            objective_summary=self._summarise_objectives(results)
        )

        self.logger.info(f"Optimisation completed in {execution_time:.2f} seconds")
        self.logger.info(f"Best score (aggregate): {self.best_score:.6f}")
        if self.best_params:
            self.logger.info(f"Best parameters: {self.best_params}")

        return search_result

    def save_results(self, result: HyperparameterSearchResult,
                     save_path: str = None) -> str:
        """Persist optimisation results to disk"""
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"hyperparameter_optimization_{timestamp}.json"
            save_path = os.path.join(cfg.experiment.results_dir, filename)

        payload = {
            'best_params': result.best_params,
            'best_score': result.best_score,
            'best_model_type': result.best_model_type,
            'best_objective_scores': result.best_objective_scores,
            'all_results': result.all_results,
            'search_history': result.search_history,
            'statistical_analysis': result.statistical_analysis,
            'execution_time': result.execution_time,
            'convergence_info': result.convergence_info,
            'objective_summary': result.objective_summary,
            'config': asdict(self.config),
            'timestamp': datetime.now().isoformat()
        }

        try:
            with open(save_path, 'w') as handle:
                json.dump(payload, handle, indent=2, default=str)
            self.logger.info(f"Hyperparameter optimisation report saved to {save_path}")
            return save_path
        except Exception as exc:
            self.logger.error(f"Failed to save optimisation results: {exc}")
            return ""

    # ------------------------------------------------------------------
    # Combination generation
    # ------------------------------------------------------------------
    def _generate_grid_search_combinations(self) -> List[Dict[str, Any]]:
        grid = self._build_parameter_grid()
        combinations: List[Dict[str, Any]] = []
        for model_type in ['user_cf', 'item_cf']:
            if model_type == 'user_cf':
                k_range = grid['user_k_neighbors']
                similarity_metrics = grid['user_similarity_metrics']
            else:
                k_range = grid['item_k_neighbors']
                similarity_metrics = grid['item_similarity_metrics']

            for params in itertools.product(
                k_range,
                similarity_metrics,
                grid['min_ratings_per_user'],
                grid['min_ratings_per_item'],
                grid['train_ratio'],
                grid['prediction_threshold']
            ):
                base = {
                    'model_type': model_type,
                    'k_neighbors': params[0],
                    'similarity_metric': params[1],
                    'min_ratings_per_user': params[2],
                    'min_ratings_per_item': params[3],
                    'train_ratio': params[4],
                    'prediction_threshold': params[5]
                }
                # Expand with regularisation / temporal options
                for shrink in grid['similarity_shrinkage_lambda']:
                    for trunc in grid['truncate_negative']:
                        if model_type.startswith('temporal'):
                            for hl in grid['temporal_half_life']:
                                for floor in grid['temporal_decay_floor']:
                                    for on_sim in grid['temporal_decay_on_similarity']:
                                        combo = base.copy()
                                        combo.update({
                                            'similarity_shrinkage_lambda': shrink,
                                            'truncate_negative': trunc,
                                            'temporal_half_life': hl,
                                            'temporal_decay_floor': floor,
                                            'temporal_decay_on_similarity': on_sim
                                        })
                                        combinations.append(combo)
                        else:
                            combo = base.copy()
                            combo.update({
                                'similarity_shrinkage_lambda': shrink,
                                'truncate_negative': trunc
                            })
                            combinations.append(combo)
        return combinations

    def _prepare_encoding_metadata(self) -> None:
        grid = self._build_parameter_grid()

        self._model_type_encoder = {'user_cf': 0.0, 'item_cf': 1.0}
        similarity_values = sorted(set(list(grid['user_similarity_metrics']) + list(grid['item_similarity_metrics'])))
        if not similarity_values:
            similarity_values = ['cosine']
        self._similarity_encoder = {metric: float(idx) for idx, metric in enumerate(similarity_values)}
        similarity_upper = float(max(len(self._similarity_encoder) - 1, 1))

        all_k = sorted(set(list(grid['user_k_neighbors']) + list(grid['item_k_neighbors']))) or [10.0]
        min_user_counts = sorted(set(grid['min_ratings_per_user'])) or [5]
        min_item_counts = sorted(set(grid['min_ratings_per_item'])) or [5]
        train_ratios = sorted(set(grid['train_ratio'])) or [0.8]
        thresholds = sorted(set(grid['prediction_threshold'])) or [3.0]

        self._normalisation_bounds = {
            'model_type': (0.0, 1.0),
            'similarity_metric': (0.0, similarity_upper),
            'k_neighbors': (float(min(all_k)), float(max(all_k))),
            'min_ratings_per_user': (float(min(min_user_counts)), float(max(min_user_counts))),
            'min_ratings_per_item': (float(min(min_item_counts)), float(max(min_item_counts))),
            'train_ratio': (float(min(train_ratios)), float(max(train_ratios))),
            'prediction_threshold': (float(min(thresholds)), float(max(thresholds)))
        }

    def _scale_value(self, key: str, value: float) -> float:
        lower, upper = self._normalisation_bounds.get(key, (0.0, 1.0))
        if upper <= lower:
            return 0.0
        return float((value - lower) / (upper - lower))

    def _encode_parameters(self, params: Dict[str, Any]) -> np.ndarray:
        model_val = self._model_type_encoder.get(params['model_type'], 0.0)
        similarity_key = params['similarity_metric']
        if similarity_key not in self._similarity_encoder:
            next_index = float(len(self._similarity_encoder))
            self._similarity_encoder[similarity_key] = next_index
            bounds = self._normalisation_bounds.get('similarity_metric', (0.0, next_index))
            self._normalisation_bounds['similarity_metric'] = (bounds[0], max(bounds[1], next_index))
        similarity_val = self._similarity_encoder[similarity_key]

        vector = np.array([
            self._scale_value('model_type', model_val),
            self._scale_value('k_neighbors', float(params['k_neighbors'])),
            self._scale_value('similarity_metric', similarity_val),
            self._scale_value('min_ratings_per_user', float(params['min_ratings_per_user'])),
            self._scale_value('min_ratings_per_item', float(params['min_ratings_per_item'])),
            self._scale_value('train_ratio', float(params['train_ratio'])),
            self._scale_value('prediction_threshold', float(params['prediction_threshold']))
        ], dtype=float)
        return vector

    def _generate_random_search_combinations(self) -> List[Dict[str, Any]]:
        grid = self._build_parameter_grid()
        combinations: List[Dict[str, Any]] = []
        rng = np.random.default_rng(self.config.cv_random_state)

        for _ in range(self.config.n_iter_random_search):
            model_type = rng.choice(['user_cf', 'item_cf'])
            if model_type == 'user_cf':
                k_neighbors = rng.choice(grid['user_k_neighbors'])
                similarity_metric = rng.choice(grid['user_similarity_metrics'])
            else:
                k_neighbors = rng.choice(grid['item_k_neighbors'])
                similarity_metric = rng.choice(grid['item_similarity_metrics'])

            params = {
                'model_type': model_type,
                'k_neighbors': int(k_neighbors),
                'similarity_metric': str(similarity_metric),
                'min_ratings_per_user': int(rng.choice(grid['min_ratings_per_user'])),
                'min_ratings_per_item': int(rng.choice(grid['min_ratings_per_item'])),
                'train_ratio': float(rng.choice(grid['train_ratio'])),
                'prediction_threshold': float(rng.choice(grid['prediction_threshold'])),
                'similarity_shrinkage_lambda': float(rng.choice(grid['similarity_shrinkage_lambda'])),
                'truncate_negative': bool(rng.choice(grid['truncate_negative']))
            }
            # Extended similarity knobs for neighborhood models
            params['similarity_significance_cap'] = int(rng.choice([50, 100]))
            params['similarity_case_amplification'] = float(rng.choice([1.1, 1.2, 1.3]))
            params['similarity_top_k'] = int(rng.choice([80, 100, 150]))
            if model_type == 'user_cf':
                params['use_iuf'] = True
            combinations.append(params)

        return combinations

    def _build_parameter_grid(self) -> Dict[str, List[Any]]:
        return {
            'user_k_neighbors': self.config.user_k_neighbors_range,
            'user_similarity_metrics': self.config.user_similarity_metrics,
            'item_k_neighbors': self.config.item_k_neighbors_range,
            'item_similarity_metrics': self.config.item_similarity_metrics,
            'min_ratings_per_user': self.config.min_ratings_per_user_range,
            'min_ratings_per_item': self.config.min_ratings_per_item_range,
            'train_ratio': self.config.train_ratio_range,
            'prediction_threshold': self.config.prediction_threshold_range,
            'similarity_shrinkage_lambda': self.config.similarity_shrinkage_lambda_values,
            'truncate_negative': self.config.truncate_negative_options,
            'temporal_half_life': self.config.temporal_half_life_range,
            'temporal_decay_floor': self.config.temporal_decay_floor_range,
            'temporal_decay_on_similarity': self.config.temporal_decay_on_similarity_options
        }

    # ------------------------------------------------------------------
    # Evaluation core
    # ------------------------------------------------------------------
    def _evaluate_parameter_combinations(self, combinations: List[Dict[str, Any]],
                                        data_loader: MovieLensLoader, user_item_matrix,
                                        timestamp_matrix=None) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        total = len(combinations)
        iterator = self._progress_wrapper(enumerate(combinations), total)

        parallel = (self.config.use_parallel_evaluation and
                    self.config.n_jobs != 1 and
                    total > 1)

        if parallel:
            executor_cls = ThreadPoolExecutor
            max_workers = None if self.config.n_jobs < 0 else self.config.n_jobs
            if self.parallel_backend == 'process':
                self.logger.warning(
                    "Process backend is experimental; falling back to thread pool for stability."
                )
            with executor_cls(max_workers=max_workers) as executor:
                futures = {}
                for idx, params in enumerate(combinations):
                    if self._should_prune(params):
                        continue
                    futures[executor.submit(
                        self._execute_trial, idx, params, data_loader, user_item_matrix, timestamp_matrix
                    )] = (idx, params)

                for future in as_completed(futures):
                    idx, params = futures[future]
                    outcome, duration = future.result()
                    if outcome:
                        result = self._finalise_result(idx, params, outcome)
                        results.append(result)
                        self._log_trial_progress(idx, total, result['mean_score'], duration)
                        continue_search = self._post_evaluation_hooks(result)
                        if not continue_search:
                            break
        else:
            for idx, params in iterator:
                if self._should_prune(params):
                    continue
                outcome, duration = self._execute_trial(idx, params, data_loader, user_item_matrix, timestamp_matrix)
                if outcome:
                    result = self._finalise_result(idx, params, outcome)
                    results.append(result)
                    self._log_trial_progress(idx, total, result['mean_score'], duration)
                    continue_search = self._post_evaluation_hooks(result)
                    if not continue_search:
                        break

                if (not self.config.enable_progress_bar and
                        (idx + 1) % self.config.progress_update_interval == 0):
                    self.logger.info(
                        f"Evaluated {idx + 1}/{total} configurations. Best score: {self.best_score:.6f}"
                    )

        return results

    def _run_bayesian_optimization(self, data_loader: MovieLensLoader, user_item_matrix,
                                   timestamp_matrix=None) -> List[Dict[str, Any]]:
        candidates = self._generate_grid_search_combinations()
        if not candidates:
            self.logger.warning("No candidate configurations generated for Bayesian optimisation. Fallback to empty result set.")
            return []

        rng = np.random.default_rng(self.config.cv_random_state)
        rng.shuffle(candidates)

        total_candidates = len(candidates)
        max_evaluations = self.config.n_iter_random_search if self.config.n_iter_random_search > 0 else total_candidates
        target_evaluations = min(total_candidates, max(self.config.n_initial_points, max_evaluations))

        evaluated_features: List[np.ndarray] = []
        evaluated_scores: List[float] = []
        results: List[Dict[str, Any]] = []
        iteration = 0
        continue_search = True

        while candidates and continue_search and len(results) < target_evaluations:
            if len(evaluated_features) < max(1, self.config.n_initial_points):
                params = candidates.pop(0)
            else:
                selected_idx = self._select_bayesian_candidate(candidates, evaluated_features, evaluated_scores)
                if selected_idx is None:
                    params = candidates.pop(0)
                else:
                    params = candidates.pop(selected_idx)

            if self._should_prune(params):
                continue

            outcome, duration = self._execute_trial(iteration, params, data_loader, user_item_matrix, timestamp_matrix)
            if not outcome:
                iteration += 1
                continue

            result = self._finalise_result(iteration, params, outcome)
            results.append(result)

            continue_search = self._post_evaluation_hooks(result)
            self._log_trial_progress(iteration, total_candidates, result['mean_score'], duration)

            evaluated_features.append(self._encode_parameters(params))
            evaluated_scores.append(result['mean_score'])
            iteration += 1

        return results

    def _select_bayesian_candidate(self, candidates: List[Dict[str, Any]],
                                   features: List[np.ndarray], scores: List[float]) -> Optional[int]:
        if len(features) < 1:
            return None

        X = np.vstack(features) if len(features) > 1 else features[0].reshape(1, -1)
        y = np.array(scores, dtype=float)

        try:
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-5)
            gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=self.config.cv_random_state)
            gp.fit(X, y)
        except Exception as exc:
            self.logger.warning(f"Bayesian surrogate failed to fit: {exc}. Falling back to random selection.")
            return None

        best_score = float(np.min(y))
        xi = 0.01
        best_ei = float('-inf')
        best_index: Optional[int] = None

        for idx, params in enumerate(candidates):
            encoded = self._encode_parameters(params).reshape(1, -1)
            mu, sigma = gp.predict(encoded, return_std=True)
            mu = float(mu[0])
            sigma = float(sigma[0])

            if sigma <= 1e-9:
                expected_improvement = 0.0
            else:
                improvement = best_score - mu - xi
                z = improvement / sigma
                expected_improvement = improvement * stats.norm.cdf(z) + sigma * stats.norm.pdf(z)

            if expected_improvement > best_ei:
                best_ei = expected_improvement
                best_index = idx

        return best_index

    def _evaluate_single_combination(self, idx: int, params: Dict[str, Any],
                                     data_loader: MovieLensLoader, user_item_matrix,
                                     timestamp_matrix=None) -> Optional[Dict[str, Any]]:
        try:
            scores, per_objective = self._cross_validate_params(
                params, data_loader, user_item_matrix, timestamp_matrix
            )
            if not scores:
                return None

            mean_score = float(np.mean(scores))
            std_score = float(np.std(scores))

            mean_objectives = self._aggregate_fold_statistics(per_objective, np.mean)
            std_objectives = self._aggregate_fold_statistics(per_objective, np.std)

            return {
                'cv_scores': scores,
                'mean_score': mean_score,
                'std_score': std_score,
                'mean_objectives': mean_objectives,
                'std_objectives': std_objectives
            }
        except Exception as exc:
            self.logger.error(f"Error evaluating parameters {params}: {exc}")
            return None

    def _execute_trial(self, idx: int, params: Dict[str, Any],
                       data_loader: MovieLensLoader, user_item_matrix,
                       timestamp_matrix=None) -> Tuple[Optional[Dict[str, Any]], float]:
        start_time = time.time()
        outcome = self._evaluate_single_combination(
            idx, params, data_loader, user_item_matrix, timestamp_matrix
        )
        return outcome, time.time() - start_time

    def _finalise_result(self, idx: int, params: Dict[str, Any], outcome: Dict[str, Any]) -> Dict[str, Any]:
        result = {
            'iteration': idx,
            'parameters': params.copy(),
            'cv_scores': outcome['cv_scores'],
            'mean_score': outcome['mean_score'],
            'std_score': outcome['std_score'],
            'mean_objectives': outcome['mean_objectives'],
            'std_objectives': outcome['std_objectives'],
            'timestamp': datetime.now().isoformat()
        }
        self.search_history.append(result)
        return result

    def _post_evaluation_hooks(self, result: Dict[str, Any]) -> bool:
        self.completed_evaluations += 1
        mean_score = result['mean_score']
        improved = self._is_better_score(mean_score, self.best_score)
        improvement_value = self._improvement_amount(mean_score, self.best_score)

        if improved and improvement_value > self.config.early_stopping_min_delta:
            self.best_score = mean_score
            self.best_iteration = result['iteration']
            self.best_params = result['parameters']
            self.best_model_type = result['parameters']['model_type']
            self.best_objective_scores = result['mean_objectives']
            self.patience_counter = 0
            self.logger.info(f"New best score: {self.best_score:.6f} with params {self.best_params}")
        else:
            self.patience_counter += 1

        continue_search = True
        if (self.config.enable_early_stopping and
                self.patience_counter >= self.config.early_stopping_patience):
            self.logger.info("Early stopping triggered by patience criterion")
            continue_search = False

        if self.config.prune_worse_than is not None and self.best_score < float('inf'):
            threshold = self.config.prune_worse_than
            threshold_value = (
                self.best_score * (1 + threshold)
                if self.config.prune_relative
                else self.best_score + threshold
            )
            if mean_score > threshold_value:
                key = (
                    result['parameters']['model_type'],
                    result['parameters']['similarity_metric'],
                    result['parameters']['k_neighbors']
                )
                self.pruned_configurations.add(key)

        if (cfg.experiment.save_intermediate_results and
                self.completed_evaluations % self.config.intermediate_save_frequency == 0):
            self._save_intermediate_results()

        return continue_search

    def _log_trial_progress(self, iteration: int, total: int, score: float, duration: float) -> None:
        percent = (iteration + 1) / max(total, 1) * 100
        best_display = self.best_score if np.isfinite(self.best_score) else score
        self.logger.info(
            f"Trial {iteration + 1}/{total} ({percent:.1f}%): score={score:.6f}, "
            f"best={best_display:.6f}, time={duration:.2f}s"
        )

    # ------------------------------------------------------------------
    # Cross validation and scoring
    # ------------------------------------------------------------------
    def _cross_validate_params(self, params: Dict[str, Any], data_loader: MovieLensLoader,
                               user_item_matrix, timestamp_matrix=None) -> Tuple[List[float], List[Dict[str, float]]]:
        ratings_data = data_loader.ratings_df
        cv_splits = self._get_cv_splits(ratings_data)
        scores: List[float] = []
        objective_scores: List[Dict[str, float]] = []

        for fold, (train_fold, val_fold) in enumerate(cv_splits):
            temp_loader = deepcopy(data_loader)
            temp_loader.ratings_df = train_fold.copy()
            temp_loader.filter_sparse_users_items(
                min_user_ratings=params['min_ratings_per_user'],
                min_item_ratings=params['min_ratings_per_item']
            )
            fold_matrix = temp_loader.create_user_item_matrix()
            fold_timestamps = temp_loader.timestamp_matrix

            try:
                model_params = {
                    'similarity_metric': params['similarity_metric'],
                    'k_neighbors': params['k_neighbors']
                }
                # Thread-through temporal where applicable
                if params.get('model_type', '').startswith('temporal'):
                    model_params.update({
                        'half_life': params.get('temporal_half_life', cfg.model.temporal_decay_half_life),
                        'decay_floor': params.get('temporal_decay_floor', cfg.model.temporal_decay_floor)
                    })

                # Apply global config knobs for similarity regularisation
                cfg.model.similarity_shrinkage_lambda = params.get('similarity_shrinkage_lambda', cfg.model.similarity_shrinkage_lambda)
                cfg.model.truncate_negative_similarity = params.get('truncate_negative', cfg.model.truncate_negative_similarity)
                cfg.model.temporal_decay_on_similarity = params.get('temporal_decay_on_similarity', cfg.model.temporal_decay_on_similarity)

                model = build_cf_model(
                    params['model_type'],
                    model_params,
                    backend=self.model_backend,
                    device=self.model_device
                )
            except Exception as exc:
                self.logger.warning(f"Skipping configuration {params} due to backend limitation: {exc}")
                scores.append(float('inf'))
                objective_scores.append({key: float('inf') for key in self.objective_keys})
                continue

            model.fit(fold_matrix, timestamp_matrix=fold_timestamps)

            fold_result = self._evaluate_model_fold(
                model, val_fold, temp_loader
            )

            scores.append(fold_result['aggregate_score'])
            objective_scores.append(fold_result['objectives'])

        return scores, objective_scores

    def _get_cv_splits(self, ratings_data: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        if self._cached_cv_splits is not None:
            return self._cached_cv_splits

        splits: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
        kf = KFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.cv_random_state
        )
        for train_idx, val_idx in kf.split(ratings_data):
            splits.append(
                (
                    ratings_data.iloc[train_idx].copy(),
                    ratings_data.iloc[val_idx].copy()
                )
            )

        self._cached_cv_splits = splits
        return self._cached_cv_splits

    def _evaluate_model_fold(self, model, val_data: pd.DataFrame,
                             data_loader: MovieLensLoader) -> Dict[str, Any]:
        true_ratings = []
        pred_ratings = []
        ranking_payload = defaultdict(list)

        for row in val_data.itertuples(index=False):
            user_id = getattr(row, 'userId')
            item_id = getattr(row, 'movieId')
            true_rating = getattr(row, 'rating')
            raw_timestamp = getattr(row, 'timestamp', None)
            normalized_timestamp = None
            if hasattr(data_loader, 'normalize_timestamp'):
                normalized_timestamp = data_loader.normalize_timestamp(raw_timestamp)

            user_idx = data_loader.user_mapping.get(user_id) if data_loader.user_mapping else None
            item_idx = data_loader.item_mapping.get(item_id) if data_loader.item_mapping else None
            if user_idx is None or item_idx is None:
                continue

            pred_rating = model.predict(user_idx, item_idx, timestamp=normalized_timestamp)
            if np.isnan(pred_rating):
                continue

            true_ratings.append(true_rating)
            pred_ratings.append(pred_rating)
            ranking_payload[user_id].append((item_id, true_rating, float(pred_rating)))

        if not true_ratings:
            penalty = float('inf')
            penalty_objectives = {key: penalty for key in self.objective_keys}
            return {'aggregate_score': penalty, 'objectives': penalty_objectives}

        objective_scores = self._compute_objective_scores(
            np.array(true_ratings), np.array(pred_ratings), ranking_payload
        )
        aggregate_score = self._aggregate_objective_scores(objective_scores)

        return {'aggregate_score': aggregate_score, 'objectives': objective_scores}

    def _compute_objective_scores(self, y_true: np.ndarray, y_pred: np.ndarray,
                                  ranking_payload: Dict[int, List[Tuple[int, float, float]]]) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        ranking_results = {}
        if ranking_payload and any(
            key in {
                OptimizationObjective.PRECISION_AT_K.value,
                OptimizationObjective.RECALL_AT_K.value,
                OptimizationObjective.NDCG_AT_K.value,
                OptimizationObjective.F1_AT_K.value
            } for key in self.objective_keys
        ):
            ranking_results = self.evaluator.evaluate_ranking(
                ranking_payload, self.k_values, self.relevance_threshold
            )

        for key in self.objective_keys:
            if key == OptimizationObjective.MAE.value:
                scores[key] = self.metrics_backend.mean_absolute_error(y_true, y_pred)
            elif key == OptimizationObjective.RMSE.value:
                scores[key] = self.metrics_backend.root_mean_squared_error(y_true, y_pred)
            elif key == OptimizationObjective.PEARSON_CORRELATION.value:
                if len(y_true) < 2:
                    scores[key] = 0.0
                else:
                    corr, _ = stats.pearsonr(y_true, y_pred)
                    scores[key] = float(corr) if not np.isnan(corr) else 0.0
            elif key == OptimizationObjective.PRECISION_AT_K.value:
                scores[key] = ranking_results.get(self.primary_k, {}).get('precision', 0.0)
            elif key == OptimizationObjective.RECALL_AT_K.value:
                scores[key] = ranking_results.get(self.primary_k, {}).get('recall', 0.0)
            elif key == OptimizationObjective.NDCG_AT_K.value:
                scores[key] = ranking_results.get(self.primary_k, {}).get('ndcg', 0.0)
            elif key == OptimizationObjective.F1_AT_K.value:
                scores[key] = ranking_results.get(self.primary_k, {}).get('f1', 0.0)
            else:
                scores[key] = float('nan')

        return scores

    def _aggregate_objective_scores(self, objective_scores: Dict[str, float]) -> float:
        total = 0.0
        for key, value in objective_scores.items():
            direction = self.config.objective_directions.get(key, False)
            if np.isnan(value):
                value = float('inf') if not direction else float('-inf')

            adjusted = -value if direction else value
            total += self.objective_weights.get(key, 1.0) * adjusted
        return total

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _resolve_objectives(self) -> List[OptimizationObjective]:
        secondary = []
        for obj in self.config.secondary_objectives or []:
            if isinstance(obj, OptimizationObjective):
                secondary.append(obj)
            else:
                secondary.append(OptimizationObjective(obj))
        ordered = [self.config.optimization_objective] + secondary

        seen = set()
        unique = []
        for obj in ordered:
            if obj not in seen:
                unique.append(obj)
                seen.add(obj)
        return unique

    def _progress_wrapper(self, iterator: Iterable, total: int) -> Iterable:
        if not self.config.enable_progress_bar:
            return iterator
        try:
            from tqdm import tqdm
        except ImportError:
            self.logger.warning(
                "tqdm is not installed; progress bar disabled."
            )
            return iterator
        return tqdm(iterator, total=total, desc="Hyperparameter Search", leave=False)

    def _should_prune(self, params: Dict[str, Any]) -> bool:
        key = (params['model_type'], params['similarity_metric'], params['k_neighbors'])
        if key in self.pruned_configurations:
            self.logger.debug(f"Skipping pruned configuration {key}")
            return True
        return False

    def _aggregate_fold_statistics(self, fold_scores: List[Dict[str, float]],
                                   reducer: Callable) -> Dict[str, float]:
        aggregated = {}
        for key in self.objective_keys:
            values = [scores.get(key, float('nan')) for scores in fold_scores]
            clean_values = [v for v in values if not np.isnan(v) and np.isfinite(v)]
            aggregated[key] = float(reducer(clean_values)) if clean_values else float('nan')
        return aggregated

    def _is_better_score(self, new_score: float, current_best: float) -> bool:
        if self.minimize_score:
            return new_score < current_best
        return new_score > current_best

    def _improvement_amount(self, new_score: float, current_best: float) -> float:
        if not np.isfinite(current_best):
            return float('inf')
        return current_best - new_score if self.minimize_score else new_score - current_best

    def _perform_statistical_analysis(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        self.logger.info("Performing statistical analysis on cross-validation scores")
        all_scores = []
        model_types = []
        for result in results:
            all_scores.extend(result['cv_scores'])
            model_types.extend([result['parameters']['model_type']] * len(result['cv_scores']))

        analysis = {
            'total_experiments': len(results),
            'total_cv_scores': len(all_scores),
            'overall_mean': float(np.mean(all_scores)) if all_scores else float('nan'),
            'overall_std': float(np.std(all_scores)) if all_scores else float('nan'),
            'overall_median': float(np.median(all_scores)) if all_scores else float('nan')
        }

        user_scores = [score for score, mtype in zip(all_scores, model_types) if mtype == 'user_cf']
        item_scores = [score for score, mtype in zip(all_scores, model_types) if mtype == 'item_cf']

        if user_scores and item_scores:
            if self.config.statistical_test_method == 'wilcoxon':
                min_len = min(len(user_scores), len(item_scores))
                if min_len > 1:
                    stat, p_value = stats.wilcoxon(user_scores[:min_len], item_scores[:min_len])
                    analysis['statistical_test'] = {
                        'method': 'wilcoxon',
                        'statistic': float(stat),
                        'p_value': float(p_value),
                        'significant': bool(p_value < self.config.significance_level)
                    }
            elif self.config.statistical_test_method == 't_test':
                stat, p_value = stats.ttest_ind(user_scores, item_scores, equal_var=False)
                analysis['statistical_test'] = {
                    'method': 't_test',
                    'statistic': float(stat),
                    'p_value': float(p_value),
                    'significant': bool(p_value < self.config.significance_level)
                }

        analysis['model_comparison'] = {
            'user_cf': {
                'count': len(user_scores),
                'mean': float(np.mean(user_scores)) if user_scores else float('nan'),
                'std': float(np.std(user_scores)) if user_scores else float('nan')
            },
            'item_cf': {
                'count': len(item_scores),
                'mean': float(np.mean(item_scores)) if item_scores else float('nan'),
                'std': float(np.std(item_scores)) if item_scores else float('nan')
            }
        }
        return analysis

    def _get_convergence_info(self) -> Dict[str, Any]:
        return {
            'best_iteration': self.best_iteration,
            'total_iterations': len(self.search_history),
            'early_stopping_triggered': (
                self.config.enable_early_stopping and
                self.patience_counter >= self.config.early_stopping_patience
            ),
            'final_patience_counter': self.patience_counter
        }

    def _save_intermediate_results(self):
        snapshot = {
            'best_score': self.best_score,
            'best_params': self.best_params,
            'evaluated_configs': len(self.search_history),
            'timestamp': datetime.now().isoformat()
        }
        try:
            with open(self.intermediate_results_path, 'w') as handle:
                json.dump(snapshot, handle, indent=2, default=str)
            self.logger.debug(
                f"Intermediate results saved to {self.intermediate_results_path}"
            )
        except Exception as exc:
            self.logger.warning(f"Failed to save intermediate snapshot: {exc}")

    def _summarise_objectives(self, results: List[Dict[str, Any]]) -> Optional[Dict[str, Dict[str, float]]]:
        if not results:
            return None
        mean_values = {}
        std_values = {}
        for key in self.objective_keys:
            values = [res['mean_objectives'].get(key, float('nan')) for res in results]
            clean_values = [v for v in values if not np.isnan(v) and np.isfinite(v)]
            if clean_values:
                mean_values[key] = float(np.mean(clean_values))
                std_values[key] = float(np.std(clean_values))
        return {'mean': mean_values, 'std': std_values}


__all__ = [
    'AcademicHyperparameterOptimizer',
    'HyperparameterSearchResult'
]
