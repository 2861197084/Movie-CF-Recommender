"""Temporal collaborative filtering models using exponential time decay."""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional, Tuple
from scipy.sparse import csr_matrix

from config import cfg
from utils.logger import get_logger
from models.user_based_cf import UserBasedCollaborativeFiltering
from models.item_based_cf import ItemBasedCollaborativeFiltering

logger = get_logger("TemporalCF")


class TemporalCollaborativeFilteringMixin:
    """Utility mixin encapsulating temporal weighting logic."""

    def __init__(self, half_life: Optional[float] = None,
                 decay_strategy: Optional[str] = None,
                 decay_floor: Optional[float] = None) -> None:
        self.temporal_half_life = float(half_life if half_life is not None else cfg.model.temporal_decay_half_life)
        self.temporal_decay_strategy = (decay_strategy or cfg.model.temporal_decay_strategy).lower()
        self.temporal_decay_floor = float(np.clip(
            decay_floor if decay_floor is not None else cfg.model.temporal_decay_floor,
            0.0,
            1.0
        ))
        self._decay_log_constant = float(np.log(2))
        self.global_latest_timestamp: float = 0.0
        self.global_earliest_timestamp: float = 0.0
        self.global_mean_timestamp: float = 0.0
        self.rating_timestamps: Dict[Tuple[int, int], float] = {}
        self.temporal_user_summary: Dict[str, np.ndarray] = {}
        self.temporal_item_summary: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Temporal statistics helpers
    # ------------------------------------------------------------------
    def _prepare_temporal_state(self, timestamp_matrix: Optional[csr_matrix],
                                shape: Tuple[int, int]) -> None:
        if timestamp_matrix is None or timestamp_matrix.nnz == 0:
            raise ValueError("Temporal collaborative filtering requires timestamp information.")

        ts_coo = timestamp_matrix.tocoo()
        data = ts_coo.data.astype(np.float64)
        rows = ts_coo.row.astype(np.int64)
        cols = ts_coo.col.astype(np.int64)

        if data.size == 0:
            raise ValueError("Timestamp matrix contains no observations.")

        self.rating_timestamps = {(int(u), int(i)): float(t) for u, i, t in zip(rows, cols, data)}
        self.global_latest_timestamp = float(np.max(data))
        self.global_earliest_timestamp = float(np.min(data))
        self.global_mean_timestamp = float(np.mean(data))

        n_users, n_items = shape
        user_latest = np.full(n_users, self.global_earliest_timestamp, dtype=np.float64)
        user_sum = np.zeros(n_users, dtype=np.float64)
        user_count = np.zeros(n_users, dtype=np.int64)

        item_latest = np.full(n_items, self.global_earliest_timestamp, dtype=np.float64)
        item_sum = np.zeros(n_items, dtype=np.float64)
        item_count = np.zeros(n_items, dtype=np.int64)

        for u, i, t in zip(rows, cols, data):
            user_sum[u] += t
            user_count[u] += 1
            if t > user_latest[u]:
                user_latest[u] = t

            item_sum[i] += t
            item_count[i] += 1
            if t > item_latest[i]:
                item_latest[i] = t

        user_mean = np.full(n_users, self.global_mean_timestamp, dtype=np.float64)
        valid_users = user_count > 0
        user_mean[valid_users] = user_sum[valid_users] / user_count[valid_users]

        item_mean = np.full(n_items, self.global_mean_timestamp, dtype=np.float64)
        valid_items = item_count > 0
        item_mean[valid_items] = item_sum[valid_items] / item_count[valid_items]

        self.temporal_user_summary = {
            'latest': user_latest,
            'mean': user_mean,
            'counts': user_count
        }
        self.temporal_item_summary = {
            'latest': item_latest,
            'mean': item_mean,
            'counts': item_count
        }

    # ------------------------------------------------------------------
    # Temporal weighting utilities
    # ------------------------------------------------------------------
    def _decay(self, delta: float) -> float:
        if self.temporal_half_life <= 0 or self.temporal_decay_strategy != 'exponential':
            return 1.0
        safe_delta = max(0.0, float(delta))
        weight = float(np.exp(-self._decay_log_constant * (safe_delta / self.temporal_half_life)))
        if self.temporal_decay_floor > 0:
            weight = max(self.temporal_decay_floor, weight)
        return weight

    def _global_decay_weight(self, rating_timestamp: Optional[float]) -> float:
        if rating_timestamp is None:
            return 1.0
        return self._decay(self.global_latest_timestamp - float(rating_timestamp))

    def _temporal_similarity_weight(self, reference_timestamp: Optional[float],
                                     rating_timestamp: Optional[float]) -> float:
        if reference_timestamp is None or rating_timestamp is None:
            return 1.0
        return self._decay(float(reference_timestamp) - float(rating_timestamp))

    def _apply_global_decay(self, interaction_matrix: csr_matrix) -> csr_matrix:
        coo = interaction_matrix.tocoo()
        weighted_values = np.zeros_like(coo.data, dtype=np.float64)
        for idx, (u, i, value) in enumerate(zip(coo.row, coo.col, coo.data)):
            ts = self.rating_timestamps.get((int(u), int(i)))
            weight = self._global_decay_weight(ts)
            weighted_values[idx] = float(value) * weight
        return csr_matrix((weighted_values, (coo.row, coo.col)), shape=interaction_matrix.shape)

    def _resolve_target_timestamp(self, user_id: int, item_id: int,
                                  provided_timestamp: Optional[float]) -> float:
        if provided_timestamp is not None:
            return float(provided_timestamp)

        candidate = None
        if self.temporal_user_summary and 'latest' in self.temporal_user_summary:
            user_latest = self.temporal_user_summary['latest']
            if 0 <= user_id < len(user_latest):
                candidate = float(user_latest[user_id])

        if (candidate is None or candidate <= 0) and self.temporal_item_summary and 'latest' in self.temporal_item_summary:
            item_latest = self.temporal_item_summary['latest']
            if 0 <= item_id < len(item_latest):
                candidate = float(item_latest[item_id])

        if candidate is None or candidate <= 0:
            candidate = self.global_latest_timestamp

        return float(candidate)

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------
    def _temporal_summary_payload(self) -> Dict[str, float]:
        return {
            'half_life': float(self.temporal_half_life),
            'decay_strategy': self.temporal_decay_strategy,
            'decay_floor': float(self.temporal_decay_floor),
            'global_latest_timestamp': float(self.global_latest_timestamp),
            'global_earliest_timestamp': float(self.global_earliest_timestamp),
            'global_mean_timestamp': float(self.global_mean_timestamp)
        }


class TemporalUserBasedCollaborativeFiltering(TemporalCollaborativeFilteringMixin,
                                              UserBasedCollaborativeFiltering):
    """User-based CF variant with exponential time-aware weighting."""

    def __init__(self, similarity_metric: str = "cosine", k_neighbors: int = 50,
                 half_life: Optional[float] = None,
                 decay_strategy: Optional[str] = None,
                 decay_floor: Optional[float] = None) -> None:
        TemporalCollaborativeFilteringMixin.__init__(self, half_life, decay_strategy, decay_floor)
        UserBasedCollaborativeFiltering.__init__(self, similarity_metric, k_neighbors)
        self.temporally_weighted_matrix: Optional[csr_matrix] = None

    def fit(self, user_item_matrix: csr_matrix,
            timestamp_matrix: Optional[csr_matrix] = None) -> 'TemporalUserBasedCollaborativeFiltering':
        logger.log_phase("Fitting Temporal User-based CF Model")
        if timestamp_matrix is None or timestamp_matrix.nnz == 0:
            raise ValueError("Temporal user-based CF requires non-empty timestamp information.")

        self.user_item_matrix = user_item_matrix
        self.timestamp_matrix = timestamp_matrix
        n_users, n_items = user_item_matrix.shape

        self._compute_user_statistics()
        self._prepare_temporal_state(timestamp_matrix, (n_users, n_items))
        # Apply optional decay on similarity construction
        if getattr(cfg.model, 'temporal_decay_on_similarity', True):
            self.temporally_weighted_matrix = self._apply_global_decay(user_item_matrix)
            base_matrix = self.temporally_weighted_matrix
        else:
            self.temporally_weighted_matrix = None
            base_matrix = user_item_matrix

        # Optional co-rating counts for shrinkage
        co_counts = None
        if base_matrix is not None:
            bin_mat = (base_matrix > 0).astype(np.float64).toarray()
            co_counts = bin_mat @ bin_mat.T

        self.similarity_matrix = self.compute_similarity_matrix(
            base_matrix,
            counts_matrix=co_counts,
            shrinkage_lambda=getattr(cfg.model, 'similarity_shrinkage_lambda', 0.0),
            truncate_negative=getattr(cfg.model, 'truncate_negative_similarity', False)
        )
        self.is_fitted = True
        logger.info("Temporal user-based CF model fitted successfully")
        return self

    def predict(self, user_id: int, item_id: int,
                timestamp: Optional[float] = None) -> float:
        self._check_fitted()

        neighbor_indices, neighbor_similarities = self.get_top_k_neighbors(user_id)
        item_ratings = self.user_item_matrix[:, item_id].toarray().flatten()

        target_time = self._resolve_target_timestamp(user_id, item_id, timestamp)

        numerator = 0.0
        denominator = 0.0

        baseline = self.user_means[user_id] if self.user_means[user_id] > 0 else self.global_mean

        for neighbor_idx, similarity in zip(neighbor_indices, neighbor_similarities):
            rating = item_ratings[neighbor_idx]
            if rating <= 0:
                continue
            neighbor_time = self.rating_timestamps.get((int(neighbor_idx), int(item_id)))
            # Causality: skip future ratings strictly after target_time
            if neighbor_time is not None and neighbor_time > target_time:
                continue
            temporal_weight = self._temporal_similarity_weight(target_time, neighbor_time)
            combined_weight = similarity * temporal_weight
            if abs(combined_weight) < 1e-12:
                continue

            neighbor_mean = self.user_means[neighbor_idx]
            numerator += combined_weight * (rating - neighbor_mean)
            denominator += abs(combined_weight)

        if denominator == 0:
            prediction = baseline
        else:
            prediction = baseline + (numerator / denominator)

        min_rating, max_rating = cfg.model.rating_scale
        return float(np.clip(prediction, min_rating, max_rating))

    def get_model_summary(self) -> Dict:
        summary = super().get_model_summary()
        summary.setdefault('temporal', {}).update(self._temporal_summary_payload())
        return summary


class TemporalItemBasedCollaborativeFiltering(TemporalCollaborativeFilteringMixin,
                                              ItemBasedCollaborativeFiltering):
    """Item-based CF variant with exponential time-aware weighting."""

    def __init__(self, similarity_metric: str = "cosine", k_neighbors: int = 50,
                 half_life: Optional[float] = None,
                 decay_strategy: Optional[str] = None,
                 decay_floor: Optional[float] = None) -> None:
        TemporalCollaborativeFilteringMixin.__init__(self, half_life, decay_strategy, decay_floor)
        ItemBasedCollaborativeFiltering.__init__(self, similarity_metric, k_neighbors)
        self.temporally_weighted_matrix: Optional[csr_matrix] = None

    def fit(self, user_item_matrix: csr_matrix,
            timestamp_matrix: Optional[csr_matrix] = None) -> 'TemporalItemBasedCollaborativeFiltering':
        logger.log_phase("Fitting Temporal Item-based CF Model")
        if timestamp_matrix is None or timestamp_matrix.nnz == 0:
            raise ValueError("Temporal item-based CF requires non-empty timestamp information.")

        self.user_item_matrix = user_item_matrix
        self.timestamp_matrix = timestamp_matrix

        n_users, n_items = user_item_matrix.shape
        self._compute_item_statistics()
        self._prepare_temporal_state(timestamp_matrix, (n_users, n_items))
        if getattr(cfg.model, 'temporal_decay_on_similarity', True):
            self.temporally_weighted_matrix = self._apply_global_decay(user_item_matrix)
            base_matrix = self.temporally_weighted_matrix
        else:
            self.temporally_weighted_matrix = None
            base_matrix = user_item_matrix

        item_user_matrix = base_matrix.T
        co_counts = None
        if item_user_matrix is not None:
            bin_mat = (item_user_matrix > 0).astype(np.float64).toarray()
            co_counts = bin_mat @ bin_mat.T

        self.similarity_matrix = self.compute_similarity_matrix(
            item_user_matrix,
            counts_matrix=co_counts,
            shrinkage_lambda=getattr(cfg.model, 'similarity_shrinkage_lambda', 0.0),
            truncate_negative=getattr(cfg.model, 'truncate_negative_similarity', False)
        )
        self.is_fitted = True
        logger.info("Temporal item-based CF model fitted successfully")
        return self

    def predict(self, user_id: int, item_id: int,
                timestamp: Optional[float] = None) -> float:
        self._check_fitted()

        neighbor_indices, neighbor_similarities = self.get_top_k_neighbors(item_id)
        user_ratings = self.user_item_matrix[user_id].toarray().flatten()

        target_time = self._resolve_target_timestamp(user_id, item_id, timestamp)

        numerator = 0.0
        denominator = 0.0

        baseline = self.item_means[item_id] if self.item_means[item_id] > 0 else self.global_mean

        for neighbor_idx, similarity in zip(neighbor_indices, neighbor_similarities):
            rating = user_ratings[neighbor_idx]
            if rating <= 0:
                continue
            neighbor_time = self.rating_timestamps.get((int(user_id), int(neighbor_idx)))
            if neighbor_time is not None and neighbor_time > target_time:
                continue
            temporal_weight = self._temporal_similarity_weight(target_time, neighbor_time)
            combined_weight = similarity * temporal_weight
            if abs(combined_weight) < 1e-12:
                continue

            neighbor_mean = self.item_means[neighbor_idx]
            numerator += combined_weight * (rating - neighbor_mean)
            denominator += abs(combined_weight)

        if denominator == 0:
            prediction = baseline
        else:
            prediction = baseline + (numerator / denominator)

        min_rating, max_rating = cfg.model.rating_scale
        return float(np.clip(prediction, min_rating, max_rating))

    def get_model_summary(self) -> Dict:
        summary = super().get_model_summary()
        summary.setdefault('temporal', {}).update(self._temporal_summary_payload())
        return summary
