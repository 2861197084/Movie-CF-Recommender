"""
User-based Collaborative Filtering Implementation
Following academic research standards and best practices
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.sparse import csr_matrix
import warnings

from models.base_cf import BaseCollaborativeFiltering
from config import cfg
from utils.logger import get_logger

logger = get_logger("UserBasedCF")

class UserBasedCollaborativeFiltering(BaseCollaborativeFiltering):
    """
    User-based Collaborative Filtering implementation

    This algorithm finds users with similar rating patterns and recommends
    items that similar users have rated highly. The approach follows the
    classical user-based CF formulation as described in academic literature.

    Mathematical Foundation:
    - Similarity computation: Various metrics (cosine, Pearson, Jaccard)
    - Prediction formula: Weighted average of similar users' ratings
    - Neighborhood selection: Top-k most similar users

    References:
    - Resnick, P., et al. (1994). GroupLens: an open architecture for collaborative filtering
    - Herlocker, J. L., et al. (1999). An algorithmic framework for performing collaborative filtering
    """

    def __init__(self, similarity_metric: str = "cosine", k_neighbors: int = 50):
        """
        Initialize User-based CF model

        Args:
            similarity_metric: Similarity metric ('cosine', 'pearson', 'jaccard')
            k_neighbors: Number of similar users to consider for predictions
        """
        super().__init__(similarity_metric, k_neighbors)
        self.user_means = None
        self.global_mean = None

    def fit(self, user_item_matrix: csr_matrix) -> 'UserBasedCollaborativeFiltering':
        """
        Fit the user-based CF model

        Args:
            user_item_matrix: Sparse user-item interaction matrix (users × items)

        Returns:
            Fitted model instance
        """
        logger.log_phase("Fitting User-based CF Model")

        self.user_item_matrix = user_item_matrix
        n_users, n_items = user_item_matrix.shape

        logger.info(f"Training on {n_users} users and {n_items} items")

        # Compute user mean ratings (for mean-centering)
        self._compute_user_statistics()

        # Compute user-user similarity matrix
        self.similarity_matrix = self.compute_similarity_matrix(user_item_matrix)

        self.is_fitted = True
        logger.info("User-based CF model fitted successfully")

        return self

    def _compute_user_statistics(self):
        """Compute user statistics for mean-centering"""
        logger.info("Computing user statistics...")

        # Convert to dense for easier computation
        dense_matrix = self.user_item_matrix.toarray()

        # Compute user means (excluding zero ratings)
        user_sums = np.sum(dense_matrix, axis=1)
        user_counts = np.sum(dense_matrix > 0, axis=1)

        # Avoid division by zero
        self.user_means = np.divide(
            user_sums, user_counts,
            out=np.zeros_like(user_sums),
            where=user_counts != 0
        )

        # Global mean rating
        total_ratings = np.sum(dense_matrix)
        total_count = np.sum(dense_matrix > 0)
        self.global_mean = total_ratings / total_count if total_count > 0 else 0.0

        logger.info(f"Computed statistics for {len(self.user_means)} users")
        logger.info(f"Global mean rating: {self.global_mean:.3f}")

    def predict(self, user_id: int, item_id: int) -> float:
        """
        Predict rating for a user-item pair using user-based CF

        Mathematical formulation:
        r̂(u,i) = r̄(u) + Σ(sim(u,v) * (r(v,i) - r̄(v))) / Σ|sim(u,v)|

        Where:
        - r̂(u,i): predicted rating for user u on item i
        - r̄(u): mean rating of user u
        - sim(u,v): similarity between users u and v
        - r(v,i): rating of user v on item i

        Args:
            user_id: User index
            item_id: Item index

        Returns:
            Predicted rating
        """
        self._check_fitted()

        # Get similar users who have rated this item
        neighbor_indices, neighbor_similarities = self.get_top_k_neighbors(user_id)

        # Filter neighbors who have rated this item
        item_ratings = self.user_item_matrix[:, item_id].toarray().flatten()
        valid_neighbors = []
        valid_similarities = []

        for neighbor_idx, similarity in zip(neighbor_indices, neighbor_similarities):
            if item_ratings[neighbor_idx] > 0:  # Neighbor has rated this item
                valid_neighbors.append(neighbor_idx)
                valid_similarities.append(similarity)

        if not valid_neighbors:
            # No similar users have rated this item
            # Return user's mean rating or global mean
            return self.user_means[user_id] if self.user_means[user_id] > 0 else self.global_mean

        valid_neighbors = np.array(valid_neighbors)
        valid_similarities = np.array(valid_similarities)

        # Apply mean-centering and weighted prediction
        user_mean = self.user_means[user_id]
        neighbor_means = self.user_means[valid_neighbors]
        neighbor_ratings = item_ratings[valid_neighbors]

        # Mean-centered ratings
        centered_ratings = neighbor_ratings - neighbor_means

        # Weighted sum
        numerator = np.sum(valid_similarities * centered_ratings)
        denominator = np.sum(np.abs(valid_similarities))

        if denominator == 0:
            prediction = user_mean if user_mean > 0 else self.global_mean
        else:
            prediction = user_mean + (numerator / denominator)

        # Clamp prediction to valid rating range
        min_rating, max_rating = cfg.model.rating_scale
        prediction = np.clip(prediction, min_rating, max_rating)

        return float(prediction)

    def predict_batch(self, user_item_pairs: List[Tuple[int, int]]) -> np.ndarray:
        """
        Batch prediction for multiple user-item pairs

        Args:
            user_item_pairs: List of (user_id, item_id) tuples

        Returns:
            Array of predicted ratings
        """
        self._check_fitted()

        predictions = np.zeros(len(user_item_pairs))

        for i, (user_id, item_id) in enumerate(user_item_pairs):
            predictions[i] = self.predict(user_id, item_id)

        return predictions

    def recommend(self, user_id: int, n_recommendations: int = 10,
                  exclude_rated: bool = True) -> List[Tuple[int, float]]:
        """
        Generate top-N recommendations for a user

        Args:
            user_id: User index
            n_recommendations: Number of recommendations to generate
            exclude_rated: Whether to exclude already rated items

        Returns:
            List of (item_id, predicted_rating) tuples sorted by rating (descending)
        """
        self._check_fitted()

        n_items = self.user_item_matrix.shape[1]
        user_ratings = self.user_item_matrix[user_id].toarray().flatten()

        # Get items to consider
        if exclude_rated:
            candidate_items = np.where(user_ratings == 0)[0]
        else:
            candidate_items = np.arange(n_items)

        if len(candidate_items) == 0:
            logger.warning(f"No candidate items for user {user_id}")
            return []

        # Predict ratings for all candidate items
        predictions = []
        for item_id in candidate_items:
            predicted_rating = self.predict(user_id, item_id)
            predictions.append((item_id, predicted_rating))

        # Sort by predicted rating (descending) and return top-N
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]

    def get_user_neighborhood(self, user_id: int, k: Optional[int] = None) -> Dict:
        """
        Get detailed information about a user's neighborhood

        Args:
            user_id: User index
            k: Number of neighbors (defaults to self.k_neighbors)

        Returns:
            Dictionary with neighborhood information
        """
        self._check_fitted()

        neighbor_indices, neighbor_similarities = self.get_top_k_neighbors(user_id, k)

        neighborhood_info = {
            "user_id": user_id,
            "user_mean_rating": self.user_means[user_id],
            "neighbors": [
                {
                    "neighbor_id": int(neighbor_id),
                    "similarity": float(similarity),
                    "neighbor_mean_rating": float(self.user_means[neighbor_id])
                }
                for neighbor_id, similarity in zip(neighbor_indices, neighbor_similarities)
            ]
        }

        return neighborhood_info

    def analyze_similarity_distribution(self) -> Dict:
        """
        Analyze the distribution of similarity scores for academic reporting

        Returns:
            Dictionary with similarity statistics
        """
        self._check_fitted()

        # Extract upper triangle (excluding diagonal)
        similarity_values = self.similarity_matrix[np.triu_indices_from(
            self.similarity_matrix, k=1)]

        stats = {
            "mean_similarity": float(np.mean(similarity_values)),
            "std_similarity": float(np.std(similarity_values)),
            "min_similarity": float(np.min(similarity_values)),
            "max_similarity": float(np.max(similarity_values)),
            "median_similarity": float(np.median(similarity_values)),
            "q25_similarity": float(np.percentile(similarity_values, 25)),
            "q75_similarity": float(np.percentile(similarity_values, 75)),
            "positive_similarities": float(np.sum(similarity_values > 0) / len(similarity_values)),
            "negative_similarities": float(np.sum(similarity_values < 0) / len(similarity_values))
        }

        return stats

    def get_model_summary(self) -> Dict:
        """Get comprehensive model summary for academic reporting"""
        base_info = self.get_model_info()

        if self.is_fitted:
            similarity_stats = self.analyze_similarity_distribution()
            base_info.update({
                "user_statistics": {
                    "global_mean_rating": float(self.global_mean),
                    "mean_user_rating": float(np.mean(self.user_means)),
                    "std_user_rating": float(np.std(self.user_means))
                },
                "similarity_statistics": similarity_stats
            })

        return base_info