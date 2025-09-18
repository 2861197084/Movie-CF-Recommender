"""
Base Collaborative Filtering class following academic research standards
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
import warnings

from config import cfg
from utils.logger import get_logger

logger = get_logger("BaseCF")

class BaseCollaborativeFiltering(ABC):
    """
    Abstract base class for collaborative filtering algorithms
    Implements common functionality and enforces consistent interface
    """

    def __init__(self, similarity_metric: str = "cosine", k_neighbors: int = 50):
        """
        Initialize base CF model

        Args:
            similarity_metric: Similarity metric to use ('cosine', 'pearson', 'jaccard')
            k_neighbors: Number of neighbors to consider for recommendations
        """
        self.similarity_metric = similarity_metric
        self.k_neighbors = k_neighbors
        self.similarity_matrix = None
        self.user_item_matrix = None
        self.is_fitted = False

        # Validate similarity metric
        valid_metrics = ['cosine', 'pearson', 'jaccard']
        if similarity_metric not in valid_metrics:
            raise ValueError(f"Similarity metric must be one of {valid_metrics}")

    @abstractmethod
    def fit(self, user_item_matrix: csr_matrix) -> 'BaseCollaborativeFiltering':
        """
        Fit the collaborative filtering model

        Args:
            user_item_matrix: Sparse user-item interaction matrix

        Returns:
            Fitted model instance
        """
        pass

    @abstractmethod
    def predict(self, user_id: int, item_id: int) -> float:
        """
        Predict rating for a user-item pair

        Args:
            user_id: User index
            item_id: Item index

        Returns:
            Predicted rating
        """
        pass

    @abstractmethod
    def recommend(self, user_id: int, n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """
        Generate top-N recommendations for a user

        Args:
            user_id: User index
            n_recommendations: Number of recommendations to generate

        Returns:
            List of (item_id, predicted_rating) tuples sorted by rating
        """
        pass

    def compute_similarity_matrix(self, data_matrix: csr_matrix) -> np.ndarray:
        """
        Compute similarity matrix between users or items

        Args:
            data_matrix: Data matrix (users × items for user-based, items × users for item-based)

        Returns:
            Similarity matrix
        """
        logger.info(f"Computing {self.similarity_metric} similarity matrix...")

        if self.similarity_metric == "cosine":
            # Convert to dense for cosine similarity computation
            dense_matrix = data_matrix.toarray()
            similarity_matrix = cosine_similarity(dense_matrix)

        elif self.similarity_metric == "pearson":
            # Pearson correlation
            similarity_matrix = self._compute_pearson_similarity(data_matrix)

        elif self.similarity_metric == "jaccard":
            # Jaccard similarity for binary interactions
            similarity_matrix = self._compute_jaccard_similarity(data_matrix)

        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")

        # Set diagonal to 0 to avoid self-similarity
        np.fill_diagonal(similarity_matrix, 0)

        logger.info(f"Similarity matrix computed: {similarity_matrix.shape}")
        return similarity_matrix

    def _compute_pearson_similarity(self, data_matrix: csr_matrix) -> np.ndarray:
        """Compute Pearson correlation similarity"""
        # Convert to dense and center the data
        dense_matrix = data_matrix.toarray().astype(np.float64)

        # Handle users/items with no ratings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            # Compute means for each row (ignoring zeros)
            row_sums = np.sum(dense_matrix, axis=1)
            row_counts = np.sum(dense_matrix > 0, axis=1)
            row_means = np.divide(row_sums, row_counts, out=np.zeros_like(row_sums), where=row_counts!=0)

            # Center the data (subtract mean only from non-zero entries)
            centered_matrix = dense_matrix.copy()
            for i in range(dense_matrix.shape[0]):
                mask = dense_matrix[i] > 0
                centered_matrix[i, mask] -= row_means[i]

            # Compute correlation
            similarity_matrix = np.corrcoef(centered_matrix)

            # Handle NaN values (users/items with constant ratings or no ratings)
            similarity_matrix = np.nan_to_num(similarity_matrix, nan=0.0)

        return similarity_matrix

    def _compute_jaccard_similarity(self, data_matrix: csr_matrix) -> np.ndarray:
        """Compute Jaccard similarity (for binary interactions)"""
        # Binarize the matrix
        binary_matrix = (data_matrix > 0).astype(np.float64)
        dense_binary = binary_matrix.toarray()

        n_entities = dense_binary.shape[0]
        similarity_matrix = np.zeros((n_entities, n_entities))

        for i in range(n_entities):
            for j in range(i, n_entities):
                intersection = np.sum(dense_binary[i] * dense_binary[j])
                union = np.sum((dense_binary[i] + dense_binary[j]) > 0)

                if union > 0:
                    jaccard_sim = intersection / union
                else:
                    jaccard_sim = 0.0

                similarity_matrix[i, j] = jaccard_sim
                similarity_matrix[j, i] = jaccard_sim

        return similarity_matrix

    def get_top_k_neighbors(self, entity_id: int, k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get top-k most similar neighbors for an entity

        Args:
            entity_id: Entity index (user or item)
            k: Number of neighbors (defaults to self.k_neighbors)

        Returns:
            Tuple of (neighbor_indices, similarity_scores)
        """
        if self.similarity_matrix is None:
            raise ValueError("Similarity matrix not computed. Call fit() first.")

        k = k or self.k_neighbors
        similarities = self.similarity_matrix[entity_id]

        # Get top-k most similar (excluding self)
        top_k_indices = np.argsort(similarities)[::-1][:k + 1]

        # Remove self if included
        if entity_id in top_k_indices:
            top_k_indices = top_k_indices[top_k_indices != entity_id][:k]
        else:
            top_k_indices = top_k_indices[:k]

        top_k_similarities = similarities[top_k_indices]

        return top_k_indices, top_k_similarities

    def _check_fitted(self):
        """Check if model has been fitted"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions. Call fit() first.")

    def get_model_info(self) -> Dict:
        """Get model information for academic reporting"""
        return {
            "model_type": self.__class__.__name__,
            "similarity_metric": self.similarity_metric,
            "k_neighbors": self.k_neighbors,
            "is_fitted": self.is_fitted,
            "matrix_shape": self.user_item_matrix.shape if self.user_item_matrix is not None else None,
            "similarity_matrix_shape": self.similarity_matrix.shape if self.similarity_matrix is not None else None
        }