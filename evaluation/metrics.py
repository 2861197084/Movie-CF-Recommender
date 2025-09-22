"""
Comprehensive evaluation metrics for collaborative filtering
Following academic research standards and best practices
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
from scipy.stats import pearsonr
import warnings

from utils.logger import get_logger

logger = get_logger("Metrics")

class RecommenderMetrics:
    """
    Comprehensive metrics suite for evaluating collaborative filtering systems

    Implements both rating prediction metrics and ranking metrics following
    academic standards in recommender systems research.

    References:
    - Herlocker, J. L., et al. (2004). Evaluating collaborative filtering recommender systems
    - Shani, G., & Gunawardana, A. (2011). Evaluating recommendation systems
    """

    @staticmethod
    def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Absolute Error (MAE)

        MAE = (1/n) * Σ|r_true - r_pred|

        Args:
            y_true: True ratings
            y_pred: Predicted ratings

        Returns:
            MAE score (lower is better)
        """
        return float(np.mean(np.abs(y_true - y_pred)))

    @staticmethod
    def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Root Mean Squared Error (RMSE)

        RMSE = sqrt((1/n) * Σ(r_true - r_pred)²)

        Args:
            y_true: True ratings
            y_pred: Predicted ratings

        Returns:
            RMSE score (lower is better)
        """
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    @staticmethod
    def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Squared Error (MSE)

        Args:
            y_true: True ratings
            y_pred: Predicted ratings

        Returns:
            MSE score (lower is better)
        """
        return float(np.mean((y_true - y_pred) ** 2))

    @staticmethod
    def normalized_mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray,
                                     rating_scale: Tuple[float, float] = (1.0, 5.0)) -> float:
        """
        Normalized Mean Absolute Error (NMAE)

        NMAE = MAE / (r_max - r_min)

        Args:
            y_true: True ratings
            y_pred: Predicted ratings
            rating_scale: Tuple of (min_rating, max_rating)

        Returns:
            NMAE score (lower is better)
        """
        mae = RecommenderMetrics.mean_absolute_error(y_true, y_pred)
        rating_range = rating_scale[1] - rating_scale[0]
        return mae / rating_range if rating_range > 0 else mae

    @staticmethod
    def precision_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int,
                       threshold: float = 3.5) -> float:
        """
        Precision@K for top-k recommendations

        Precision@K = |{relevant items} ∩ {recommended items}| / k

        Args:
            y_true: True ratings
            y_pred: Predicted ratings (sorted by relevance)
            k: Number of top recommendations to consider
            threshold: Threshold for considering an item relevant

        Returns:
            Precision@K score [0, 1] (higher is better)
        """
        if len(y_pred) < k:
            k = len(y_pred)

        if k == 0:
            return 0.0

        # Get top-k predictions
        top_k_true = y_true[:k]

        # Count relevant items in top-k
        relevant_count = np.sum(top_k_true >= threshold)

        return float(relevant_count / k)

    @staticmethod
    def recall_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int,
                    threshold: float = 3.5) -> float:
        """
        Recall@K for top-k recommendations

        Recall@K = |{relevant items} ∩ {recommended items}| / |{relevant items}|

        Args:
            y_true: True ratings
            y_pred: Predicted ratings (sorted by relevance)
            k: Number of top recommendations to consider
            threshold: Threshold for considering an item relevant

        Returns:
            Recall@K score [0, 1] (higher is better)
        """
        # Total relevant items
        total_relevant = np.sum(y_true >= threshold)

        if total_relevant == 0:
            return 0.0

        if len(y_pred) < k:
            k = len(y_pred)

        if k == 0:
            return 0.0

        # Get top-k predictions
        top_k_true = y_true[:k]

        # Count relevant items in top-k
        relevant_in_topk = np.sum(top_k_true >= threshold)

        return float(relevant_in_topk / total_relevant)

    @staticmethod
    def f1_score_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int,
                      threshold: float = 3.5) -> float:
        """
        F1-Score@K for top-k recommendations

        F1@K = 2 * (Precision@K * Recall@K) / (Precision@K + Recall@K)

        Args:
            y_true: True ratings
            y_pred: Predicted ratings (sorted by relevance)
            k: Number of top recommendations to consider
            threshold: Threshold for considering an item relevant

        Returns:
            F1@K score [0, 1] (higher is better)
        """
        precision = RecommenderMetrics.precision_at_k(y_true, y_pred, k, threshold)
        recall = RecommenderMetrics.recall_at_k(y_true, y_pred, k, threshold)

        if precision + recall == 0:
            return 0.0

        return float(2 * precision * recall / (precision + recall))

    @staticmethod
    def dcg_at_k(y_true: np.ndarray, k: int) -> float:
        """
        Discounted Cumulative Gain at K

        DCG@K = Σ(i=1 to k) (2^rel_i - 1) / log2(i + 1)

        Args:
            y_true: True ratings (sorted by predicted relevance)
            k: Number of top recommendations to consider

        Returns:
            DCG@K score (higher is better)
        """
        if len(y_true) < k:
            k = len(y_true)

        if k == 0:
            return 0.0

        # Get top-k ratings
        top_k_ratings = y_true[:k]

        # Calculate DCG
        dcg = 0.0
        for i, rating in enumerate(top_k_ratings):
            # Using 2^rating - 1 for relevance score
            relevance = 2 ** rating - 1
            discount = np.log2(i + 2)  # i+2 because we start from position 1
            dcg += relevance / discount

        return float(dcg)

    @staticmethod
    def ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
        """
        Normalized Discounted Cumulative Gain at K

        NDCG@K = DCG@K / IDCG@K

        Args:
            y_true: True ratings
            y_pred: Predicted ratings (used for ranking)
            k: Number of top recommendations to consider

        Returns:
            NDCG@K score [0, 1] (higher is better)
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have same length")

        if len(y_pred) < k:
            k = len(y_pred)

        if k == 0:
            return 0.0

        # Sort true ratings by predicted scores (descending)
        sorted_indices = np.argsort(y_pred)[::-1]
        sorted_true_ratings = y_true[sorted_indices]

        # Calculate DCG@K
        dcg = RecommenderMetrics.dcg_at_k(sorted_true_ratings, k)

        # Calculate IDCG@K (ideal DCG with perfect ranking)
        ideal_sorted_ratings = np.sort(y_true)[::-1]  # Sort true ratings descending
        idcg = RecommenderMetrics.dcg_at_k(ideal_sorted_ratings, k)

        if idcg == 0:
            return 0.0

        return float(dcg / idcg)

    @staticmethod
    def hit_rate_at_k(y_true: np.ndarray, k: int, threshold: float = 3.5) -> float:
        """
        Hit Rate@K (also known as Recall@K for binary relevance)

        Hit Rate@K = 1 if any item in top-k is relevant, 0 otherwise

        Args:
            y_true: True ratings (sorted by predicted relevance)
            k: Number of top recommendations to consider
            threshold: Threshold for considering an item relevant

        Returns:
            Hit Rate@K score [0, 1] (higher is better)
        """
        if len(y_true) < k:
            k = len(y_true)

        if k == 0:
            return 0.0

        top_k_ratings = y_true[:k]
        return float(1.0 if np.any(top_k_ratings >= threshold) else 0.0)

    @staticmethod
    def average_reciprocal_rank(y_true_list: List[np.ndarray],
                               threshold: float = 3.5) -> float:
        """
        Average Reciprocal Rank (ARR)

        ARR = (1/|Q|) * Σ(1/rank of first relevant item)

        Args:
            y_true_list: List of true rating arrays (each sorted by predicted relevance)
            threshold: Threshold for considering an item relevant

        Returns:
            ARR score (higher is better)
        """
        if not y_true_list:
            return 0.0

        reciprocal_ranks = []

        for y_true in y_true_list:
            # Find first relevant item
            relevant_positions = np.where(y_true >= threshold)[0]

            if len(relevant_positions) > 0:
                # Position is 1-indexed
                first_relevant_rank = relevant_positions[0] + 1
                reciprocal_ranks.append(1.0 / first_relevant_rank)
            else:
                reciprocal_ranks.append(0.0)

        return float(np.mean(reciprocal_ranks))

    @staticmethod
    def coverage(recommended_items: List[List[int]], total_items: int) -> float:
        """
        Catalog Coverage - fraction of items that appear in recommendations

        Coverage = |{recommended items}| / |{all items}|

        Args:
            recommended_items: List of recommendation lists for different users
            total_items: Total number of items in catalog

        Returns:
            Coverage score [0, 1] (higher indicates better coverage)
        """
        if total_items == 0:
            return 0.0

        # Get all unique recommended items
        all_recommended = set()
        for user_recommendations in recommended_items:
            all_recommended.update(user_recommendations)

        return float(len(all_recommended) / total_items)

    @staticmethod
    def novelty(recommended_items: List[List[int]], item_popularity: Dict[int, float]) -> float:
        """
        Average novelty of recommendations

        Novelty = -log2(popularity)

        Args:
            recommended_items: List of recommendation lists for different users
            item_popularity: Dictionary mapping item_id to popularity score [0, 1]

        Returns:
            Average novelty score (higher indicates more novel recommendations)
        """
        if not recommended_items:
            return 0.0

        novelty_scores = []

        for user_recommendations in recommended_items:
            user_novelty = []
            for item_id in user_recommendations:
                popularity = item_popularity.get(item_id, 0.001)  # Avoid log(0)
                novelty = -np.log2(max(popularity, 1e-10))  # Avoid log(0)
                user_novelty.append(novelty)

            if user_novelty:
                novelty_scores.append(np.mean(user_novelty))

        return float(np.mean(novelty_scores)) if novelty_scores else 0.0

class MetricsEvaluator:
    """
    Comprehensive evaluator for collaborative filtering models
    """

    def __init__(self, metrics_config: Optional[Dict] = None):
        """
        Initialize metrics evaluator

        Args:
            metrics_config: Configuration for metrics computation
        """
        self.config = metrics_config or {}
        self.metrics = RecommenderMetrics()

    def evaluate_rating_prediction(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 rating_scale: Tuple[float, float] = (1.0, 5.0)) -> Dict[str, float]:
        """
        Evaluate rating prediction performance

        Args:
            y_true: True ratings
            y_pred: Predicted ratings
            rating_scale: Rating scale (min, max)

        Returns:
            Dictionary of metric scores
        """
        logger.info("Evaluating rating prediction metrics...")

        # Ensure arrays are 1D
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()

        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have same length")

        results = {}

        # Basic metrics
        results['mae'] = self.metrics.mean_absolute_error(y_true, y_pred)
        results['rmse'] = self.metrics.root_mean_squared_error(y_true, y_pred)
        results['mse'] = self.metrics.mean_squared_error(y_true, y_pred)
        results['nmae'] = self.metrics.normalized_mean_absolute_error(y_true, y_pred, rating_scale)

        # Correlation metrics
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            correlation, p_value = pearsonr(y_true, y_pred)
            results['pearson_correlation'] = float(correlation) if not np.isnan(correlation) else 0.0
            results['pearson_p_value'] = float(p_value) if not np.isnan(p_value) else 1.0

        # Additional statistics
        results['mean_true_rating'] = float(np.mean(y_true))
        results['mean_pred_rating'] = float(np.mean(y_pred))
        results['std_true_rating'] = float(np.std(y_true))
        results['std_pred_rating'] = float(np.std(y_pred))

        logger.info(f"Rating prediction evaluation completed. MAE: {results['mae']:.4f}, RMSE: {results['rmse']:.4f}")

        return results

    def evaluate_ranking(self, user_item_predictions: Dict[int, List[Tuple[int, float, float]]],
                        k_values: List[int] = [5, 10, 20], threshold: float = 3.5) -> Dict[str, Dict[int, float]]:
        """
        Evaluate ranking performance

        Args:
            user_item_predictions: Dict mapping user_id to list of (item_id, true_rating, pred_rating)
            k_values: List of k values to evaluate
            threshold: Threshold for relevance

        Returns:
            Dictionary of metric scores for each k value
        """
        logger.info("Evaluating ranking metrics...")

        results = {k: {} for k in k_values}

        for k in k_values:
            precision_scores = []
            recall_scores = []
            f1_scores = []
            ndcg_scores = []
            hit_rates = []

            for user_id, predictions in user_item_predictions.items():
                if not predictions:
                    continue

                # Sort by predicted rating (descending)
                sorted_predictions = sorted(predictions, key=lambda x: x[2], reverse=True)

                # Extract true and predicted ratings
                true_ratings = np.array([x[1] for x in sorted_predictions])
                pred_ratings = np.array([x[2] for x in sorted_predictions])

                # Calculate metrics
                precision_scores.append(self.metrics.precision_at_k(true_ratings, pred_ratings, k, threshold))
                recall_scores.append(self.metrics.recall_at_k(true_ratings, pred_ratings, k, threshold))
                f1_scores.append(self.metrics.f1_score_at_k(true_ratings, pred_ratings, k, threshold))
                ndcg_scores.append(self.metrics.ndcg_at_k(true_ratings, pred_ratings, k))
                hit_rates.append(self.metrics.hit_rate_at_k(true_ratings, k, threshold))

            # Average across users
            results[k]['precision'] = float(np.mean(precision_scores)) if precision_scores else 0.0
            results[k]['recall'] = float(np.mean(recall_scores)) if recall_scores else 0.0
            results[k]['f1'] = float(np.mean(f1_scores)) if f1_scores else 0.0
            results[k]['ndcg'] = float(np.mean(ndcg_scores)) if ndcg_scores else 0.0
            results[k]['hit_rate'] = float(np.mean(hit_rates)) if hit_rates else 0.0

        logger.info("Ranking evaluation completed")
        return results

    def evaluate_diversity_and_novelty(self, user_recommendations: Dict[int, List[int]],
                                     item_popularity: Dict[int, float],
                                     total_items: int) -> Dict[str, float]:
        """
        Evaluate diversity and novelty metrics

        Args:
            user_recommendations: Dict mapping user_id to list of recommended item_ids
            item_popularity: Dict mapping item_id to popularity score
            total_items: Total number of items in catalog

        Returns:
            Dictionary of diversity/novelty metrics
        """
        logger.info("Evaluating diversity and novelty metrics...")

        recommendation_lists = list(user_recommendations.values())

        results = {}
        results['coverage'] = self.metrics.coverage(recommendation_lists, total_items)
        results['novelty'] = self.metrics.novelty(recommendation_lists, item_popularity)

        logger.info("Diversity and novelty evaluation completed")
        return results

    def comprehensive_evaluation(self, model, test_data: pd.DataFrame,
                               user_item_matrix,
                               k_values: List[int] = [5, 10, 20],
                               threshold: float = 3.5,
                               data_loader=None,
                               train_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Perform comprehensive evaluation of a collaborative filtering model

        Args:
            model: Fitted CF model
            test_data: Test dataset with columns ['userId', 'movieId', 'rating']
            user_item_matrix: User-item interaction matrix
            k_values: List of k values for ranking evaluation
            threshold: Relevance threshold
            data_loader: DataLoader instance with user/item mappings
            train_data: Optional training dataframe used for popularity statistics

        Returns:
            Comprehensive evaluation results
        """
        logger.log_phase("Comprehensive Model Evaluation")

        results = {
            'rating_prediction': {},
            'ranking': {},
            'ranking_summary': {},
            'diversity_novelty': {},
            'model_info': model.get_model_summary() if hasattr(model, 'get_model_summary') else {}
        }

        # Get user and item mappings
        if hasattr(model, 'user_item_matrix') and hasattr(data_loader, 'user_mapping'):
            user_mapping = data_loader.user_mapping
            item_mapping = data_loader.item_mapping
        else:
            logger.warning("No user/item mappings found. Creating mappings from test data...")
            unique_users = sorted(test_data['userId'].unique())
            unique_items = sorted(test_data['movieId'].unique())
            user_mapping = {user_id: idx for idx, user_id in enumerate(unique_users)}
            item_mapping = {item_id: idx for idx, item_id in enumerate(unique_items)}

        # Rating prediction evaluation
        logger.info("Computing rating predictions...")
        y_true: List[float] = []
        y_pred: List[float] = []
        prediction_cache: Dict[Tuple[int, int], Tuple[float, Optional[float], Optional[float]]] = {}
        detailed_records: List[Dict[str, Optional[float]]] = []
        max_detailed_records = 5000

        normalize_timestamp = None
        if data_loader is not None and hasattr(data_loader, 'normalize_timestamp'):
            normalize_timestamp = data_loader.normalize_timestamp  # type: ignore[attr-defined]

        skipped_total = 0
        skipped_missing_user = 0
        skipped_missing_item = 0
        for _, row in test_data.iterrows():
            original_user_id = row['userId']
            original_item_id = row['movieId']
            true_rating = row['rating']

            user_idx = user_mapping.get(original_user_id)
            item_idx = item_mapping.get(original_item_id)

            if user_idx is None or item_idx is None:
                skipped_total += 1
                if user_idx is None:
                    skipped_missing_user += 1
                if item_idx is None:
                    skipped_missing_item += 1
                continue

            try:
                raw_timestamp = row['timestamp'] if 'timestamp' in row else None
                if pd.isna(raw_timestamp):
                    raw_timestamp = None
                normalized_timestamp = None
                if normalize_timestamp is not None:
                    normalized_timestamp = normalize_timestamp(raw_timestamp)

                pred_rating = model.predict(user_idx, item_idx, timestamp=normalized_timestamp)
                y_true.append(float(true_rating))
                y_pred.append(float(pred_rating))
                prediction_cache[(user_idx, item_idx)] = (
                    float(pred_rating),
                    float(normalized_timestamp) if normalized_timestamp is not None else None,
                    float(raw_timestamp) if raw_timestamp is not None else None
                )
                if len(detailed_records) < max_detailed_records:
                    detailed_records.append({
                        'user_id': float(original_user_id),
                        'item_id': float(original_item_id),
                        'true_rating': float(true_rating),
                        'pred_rating': float(pred_rating),
                        'timestamp': float(raw_timestamp) if raw_timestamp is not None else None,
                        'normalized_timestamp': float(normalized_timestamp) if normalized_timestamp is not None else None
                    })
            except Exception as e:
                logger.warning(f"Prediction failed for user {original_user_id} (idx {user_idx}), item {original_item_id} (idx {item_idx}): {e}")
                continue

        if y_true and y_pred:
            results['rating_prediction'] = self.evaluate_rating_prediction(
                np.array(y_true), np.array(y_pred)
            )

        if skipped_total > 0:
            logger.info(
                f"Skipped {skipped_total} test pairs due to unseen entities (users: {skipped_missing_user}, items: {skipped_missing_item})."
            )
        logger.info("Evaluating ranking performance...")
        unique_users = test_data['userId'].unique()
        sample_users = np.random.choice(unique_users, size=min(100, len(unique_users)), replace=False)

        user_item_predictions: Dict[int, List[Tuple[int, float, float]]] = {}
        for original_user_id in sample_users:
            user_test_data = test_data[test_data['userId'] == original_user_id]
            if len(user_test_data) < 5:
                continue

            user_idx = user_mapping.get(original_user_id)
            if user_idx is None:
                continue

            predictions: List[Tuple[int, float, float]] = []
            for _, row in user_test_data.iterrows():
                original_item_id = row['movieId']
                true_rating = row['rating']
                item_idx = item_mapping.get(original_item_id)
                if item_idx is None:
                    continue
                try:
                    cached = prediction_cache.get((user_idx, item_idx))
                    if cached is not None:
                        pred_rating = cached[0]
                    else:
                        raw_timestamp = row['timestamp'] if 'timestamp' in row else None
                        if pd.isna(raw_timestamp):
                            raw_timestamp = None
                        normalized_timestamp = None
                        if normalize_timestamp is not None:
                            normalized_timestamp = normalize_timestamp(raw_timestamp)
                        pred_rating = model.predict(user_idx, item_idx, timestamp=normalized_timestamp)
                        prediction_cache[(user_idx, item_idx)] = (
                            float(pred_rating),
                            float(normalized_timestamp) if normalized_timestamp is not None else None,
                            float(raw_timestamp) if raw_timestamp is not None else None
                        )
                    predictions.append((original_item_id, float(true_rating), float(pred_rating)))
                except Exception:
                    continue

            if predictions:
                user_item_predictions[original_user_id] = predictions

        ranking_user_lists: Dict[int, List[int]] = {}
        arr_payload: List[np.ndarray] = []

        if user_item_predictions:
            results['ranking'] = self.evaluate_ranking(user_item_predictions, k_values, threshold)

            max_k = max(k_values) if k_values else 0
            for user_id, predictions in user_item_predictions.items():
                sorted_predictions = sorted(predictions, key=lambda x: x[2], reverse=True)
                true_ratings_sorted = np.array([p[1] for p in sorted_predictions])
                arr_payload.append(true_ratings_sorted)

                if max_k > 0:
                    ranking_user_lists[user_id] = [int(p[0]) for p in sorted_predictions[:max_k]]

            if arr_payload:
                results['ranking_summary']['mean_reciprocal_rank'] = self.metrics.average_reciprocal_rank(
                    arr_payload,
                    threshold=threshold
                )

        if ranking_user_lists:
            source_df = None
            if train_data is not None and not train_data.empty:
                source_df = train_data
            elif data_loader is not None and getattr(data_loader, 'ratings_df', None) is not None:
                source_df = data_loader.ratings_df

            item_popularity: Optional[Dict[int, float]] = None
            total_items = len(item_mapping) if item_mapping is not None else 0

            if source_df is not None:
                item_counts = source_df['movieId'].value_counts()
                total_count = float(item_counts.sum())
                if total_count > 0:
                    item_popularity = {int(item): count / total_count for item, count in item_counts.items()}
                    if total_items == 0:
                        total_items = len(item_popularity)

            if item_popularity:
                results['diversity_novelty'] = self.evaluate_diversity_and_novelty(
                    ranking_user_lists,
                    item_popularity,
                    total_items
                )

        if detailed_records:
            results['prediction_details'] = detailed_records

        if not results['ranking_summary']:
            results.pop('ranking_summary')
        if not results['diversity_novelty']:
            results.pop('diversity_novelty')

        logger.info("Comprehensive evaluation completed")
        return results
