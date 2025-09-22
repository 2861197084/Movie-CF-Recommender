"""
Item-based Collaborative Filtering Implementation
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

logger = get_logger("ItemBasedCF")

class ItemBasedCollaborativeFiltering(BaseCollaborativeFiltering):
    """
    Item-based Collaborative Filtering implementation

    This algorithm finds items with similar rating patterns and recommends
    items that are similar to items the user has rated highly. The approach
    is based on the assumption that users will prefer items similar to those
    they have liked before.

    Mathematical Foundation:
    - Item similarity computation: Various metrics (cosine, Pearson, Jaccard)
    - Prediction formula: Weighted average of similar items' ratings
    - Neighborhood selection: Top-k most similar items

    References:
    - Sarwar, B., et al. (2001). Item-based collaborative filtering recommendation algorithms
    - Deshpande, M., & Karypis, G. (2004). Item-based top-n recommendation algorithms
    """

    def __init__(self, similarity_metric: str = "cosine", k_neighbors: int = 50):
        """
        Initialize Item-based CF model

        Args:
            similarity_metric: Similarity metric ('cosine', 'pearson', 'jaccard')
            k_neighbors: Number of similar items to consider for predictions
        """
        super().__init__(similarity_metric, k_neighbors)
        self.item_means = None
        self.global_mean = None

    def fit(self, user_item_matrix: csr_matrix,
            timestamp_matrix: Optional[csr_matrix] = None) -> 'ItemBasedCollaborativeFiltering':
        """
        Fit the item-based CF model

        Args:
            user_item_matrix: Sparse user-item interaction matrix (users × items)
            timestamp_matrix: Optional timestamp matrix aligned with ratings

        Returns:
            Fitted model instance
        """
        logger.log_phase("Fitting Item-based CF Model")

        self.user_item_matrix = user_item_matrix
        self.timestamp_matrix = timestamp_matrix
        n_users, n_items = user_item_matrix.shape

        logger.info(f"Training on {n_users} users and {n_items} items")

        # Compute item mean ratings (for mean-centering)
        self._compute_item_statistics()

        # Compute item-item similarity matrix
        # Transpose the matrix to get items × users for similarity computation
        item_user_matrix = user_item_matrix.T
        self.similarity_matrix = self.compute_similarity_matrix(item_user_matrix)

        self.is_fitted = True
        logger.info("Item-based CF model fitted successfully")

        return self

    def _compute_item_statistics(self):
        """Compute item statistics for mean-centering"""
        logger.info("Computing item statistics...")

        # Convert to dense for easier computation
        dense_matrix = self.user_item_matrix.toarray()

        # Compute item means (excluding zero ratings)
        item_sums = np.sum(dense_matrix, axis=0)  # Sum over users
        item_counts = np.sum(dense_matrix > 0, axis=0)  # Count non-zero ratings

        # Avoid division by zero
        self.item_means = np.divide(
            item_sums, item_counts,
            out=np.zeros_like(item_sums),
            where=item_counts != 0
        )

        # Global mean rating
        total_ratings = np.sum(dense_matrix)
        total_count = np.sum(dense_matrix > 0)
        self.global_mean = total_ratings / total_count if total_count > 0 else 0.0

        logger.info(f"Computed statistics for {len(self.item_means)} items")
        logger.info(f"Global mean rating: {self.global_mean:.3f}")

    def predict(self, user_id: int, item_id: int,
                timestamp: Optional[float] = None) -> float:
        """
        Predict rating for a user-item pair using item-based CF

        Mathematical formulation:
        r̂(u,i) = r̄(i) + Σ(sim(i,j) * (r(u,j) - r̄(j))) / Σ|sim(i,j)|

        Where:
        - r̂(u,i): predicted rating for user u on item i
        - r̄(i): mean rating of item i
        - sim(i,j): similarity between items i and j
        - r(u,j): rating of user u on item j

        Args:
            user_id: User index
            item_id: Item index

        Returns:
            Predicted rating
        """
        self._check_fitted()

        # Get similar items that the user has rated
        neighbor_indices, neighbor_similarities = self.get_top_k_neighbors(item_id)

        # Get user's ratings
        user_ratings = self.user_item_matrix[user_id].toarray().flatten()

        # Filter neighbors that the user has rated
        valid_neighbors = []
        valid_similarities = []

        for neighbor_idx, similarity in zip(neighbor_indices, neighbor_similarities):
            if user_ratings[neighbor_idx] > 0:  # User has rated this item
                valid_neighbors.append(neighbor_idx)
                valid_similarities.append(similarity)

        if not valid_neighbors:
            # User hasn't rated any similar items
            # Return item's mean rating or global mean
            return self.item_means[item_id] if self.item_means[item_id] > 0 else self.global_mean

        valid_neighbors = np.array(valid_neighbors)
        valid_similarities = np.array(valid_similarities)

        # Apply mean-centering and weighted prediction
        item_mean = self.item_means[item_id]
        neighbor_means = self.item_means[valid_neighbors]
        neighbor_ratings = user_ratings[valid_neighbors]

        # Mean-centered ratings
        centered_ratings = neighbor_ratings - neighbor_means

        # Weighted sum
        numerator = np.sum(valid_similarities * centered_ratings)
        denominator = np.sum(np.abs(valid_similarities))

        if denominator == 0:
            prediction = item_mean if item_mean > 0 else self.global_mean
        else:
            prediction = item_mean + (numerator / denominator)

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

        Strategy:
        1. For each unrated item, predict the rating based on similar items
        2. Sort by predicted rating and return top-N

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

    def get_item_neighborhood(self, item_id: int, k: Optional[int] = None) -> Dict:
        """
        Get detailed information about an item's neighborhood

        Args:
            item_id: Item index
            k: Number of neighbors (defaults to self.k_neighbors)

        Returns:
            Dictionary with neighborhood information
        """
        self._check_fitted()

        neighbor_indices, neighbor_similarities = self.get_top_k_neighbors(item_id, k)

        neighborhood_info = {
            "item_id": item_id,
            "item_mean_rating": self.item_means[item_id],
            "neighbors": [
                {
                    "neighbor_id": int(neighbor_id),
                    "similarity": float(similarity),
                    "neighbor_mean_rating": float(self.item_means[neighbor_id])
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

    def find_similar_items(self, item_id: int, n_similar: int = 10) -> List[Tuple[int, float]]:
        """
        Find most similar items to a given item

        Args:
            item_id: Target item index
            n_similar: Number of similar items to return

        Returns:
            List of (similar_item_id, similarity_score) tuples
        """
        self._check_fitted()

        neighbor_indices, neighbor_similarities = self.get_top_k_neighbors(item_id, n_similar)

        similar_items = [
            (int(item_idx), float(similarity))
            for item_idx, similarity in zip(neighbor_indices, neighbor_similarities)
        ]

        return similar_items

    def get_content_based_features(self, item_id: int) -> Dict:
        """
        Extract features that could be used for content-based analysis

        Args:
            item_id: Item index

        Returns:
            Dictionary with item features
        """
        self._check_fitted()

        # Get item's rating distribution
        item_ratings = self.user_item_matrix[:, item_id].toarray().flatten()
        item_ratings = item_ratings[item_ratings > 0]  # Remove zeros

        if len(item_ratings) == 0:
            return {"item_id": item_id, "n_ratings": 0}

        features = {
            "item_id": item_id,
            "n_ratings": len(item_ratings),
            "mean_rating": float(np.mean(item_ratings)),
            "std_rating": float(np.std(item_ratings)),
            "min_rating": float(np.min(item_ratings)),
            "max_rating": float(np.max(item_ratings)),
            "rating_distribution": {
                str(rating): int(count)
                for rating, count in zip(*np.unique(item_ratings, return_counts=True))
            }
        }

        return features

    def get_model_summary(self) -> Dict:
        """Get comprehensive model summary for academic reporting"""
        base_info = self.get_model_info()

        if self.is_fitted:
            similarity_stats = self.analyze_similarity_distribution()
            base_info.update({
                "item_statistics": {
                    "global_mean_rating": float(self.global_mean),
                    "mean_item_rating": float(np.mean(self.item_means)),
                    "std_item_rating": float(np.std(self.item_means))
                },
                "similarity_statistics": similarity_stats
            })

        return base_info