"""PyTorch-based collaborative filtering models with GPU support."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False

if TYPE_CHECKING:  # pragma: no cover
    import torch as torch_types  # noqa: F401

from config import cfg
from utils.logger import get_logger


logger = get_logger("TorchCF")


def ensure_torch_available() -> None:
    """Raise an informative error when torch backend is requested but unavailable."""
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch backend requested but torch is not installed.")


def _resolve_device(requested: Optional[str]) -> "torch.device":
    ensure_torch_available()

    # Auto-detect best available device
    if requested is None:
        if torch.cuda.is_available():
            default_device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            default_device = "mps"
        else:
            default_device = "cpu"
    else:
        default_device = requested

    target = default_device

    # Validate device availability
    if target == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        target = "cpu"
    elif target == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            logger.warning("MPS (Metal Performance Shaders) requested but not available, falling back to CPU")
            target = "cpu"

    device_obj = torch.device(target)
    logger.info(f"Using device: {device_obj}")
    return device_obj


def _to_tensor(matrix: Any, device: "torch.device") -> "torch.Tensor":
    if torch.is_tensor(matrix):
        return matrix.to(device=device, dtype=torch.float32)
    if hasattr(matrix, "toarray"):
        dense = matrix.toarray().astype(np.float32, copy=False)
    else:
        dense = np.asarray(matrix, dtype=np.float32)
    return torch.from_numpy(dense).to(device)


def _mean_ignore_zero(tensor: "torch.Tensor", dim: int) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    mask = (tensor > 0).float()
    counts = mask.sum(dim=dim)
    sums = (tensor * mask).sum(dim=dim)
    means = torch.zeros_like(sums)
    nonzero = counts > 0
    means[nonzero] = sums[nonzero] / counts[nonzero]
    return means, mask, counts


def _clamp_rating(value: "torch.Tensor") -> float:
    min_rating, max_rating = cfg.model.rating_scale
    return float(torch.clamp(value, min=float(min_rating), max=float(max_rating)).item())


def _build_sparse_topk(
    similarity: "torch.Tensor",
    k: int,
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Construct sparse top-k similarity matrices and return top-k metadata."""
    n_entities = similarity.size(0)
    if n_entities == 0:
        raise ValueError("Similarity matrix must contain at least one entity")

    if n_entities == 1:
        empty_indices = torch.zeros((2, 0), dtype=torch.long, device=similarity.device)
        empty_values = torch.zeros(0, dtype=similarity.dtype, device=similarity.device)
        sparse = torch.sparse_coo_tensor(empty_indices, empty_values, size=(1, 1), device=similarity.device).coalesce()
        return sparse, sparse.clone(), torch.empty((1, 0), device=similarity.device), torch.empty((1, 0), dtype=torch.long, device=similarity.device)

    k = max(0, min(k, n_entities - 1))
    if k == 0:
        empty_indices = torch.zeros((2, 0), dtype=torch.long, device=similarity.device)
        empty_values = torch.zeros(0, dtype=similarity.dtype, device=similarity.device)
        sparse = torch.sparse_coo_tensor(empty_indices, empty_values, size=similarity.shape, device=similarity.device).coalesce()
        return sparse, sparse.clone(), torch.empty((n_entities, 0), device=similarity.device), torch.empty((n_entities, 0), dtype=torch.long, device=similarity.device)

    diag_cache = similarity.diagonal().clone()
    similarity.fill_diagonal_(float('-inf'))
    topk_values, topk_indices = torch.topk(similarity, k=k, dim=1)
    similarity.diagonal().copy_(diag_cache)

    row_indices = torch.arange(n_entities, device=similarity.device).unsqueeze(1).expand(-1, k).reshape(-1)
    col_indices = topk_indices.reshape(-1)
    flat_values = topk_values.reshape(-1)

    valid = col_indices >= 0
    row_indices = row_indices[valid]
    col_indices = col_indices[valid]
    flat_values = flat_values[valid]

    indices = torch.stack([row_indices, col_indices], dim=0)
    sparse_topk = torch.sparse_coo_tensor(indices, flat_values, size=similarity.shape, device=similarity.device).coalesce()
    sparse_abs = torch.sparse_coo_tensor(indices, flat_values.abs(), size=similarity.shape, device=similarity.device).coalesce()

    return sparse_topk, sparse_abs, topk_values, topk_indices


@dataclass
class TorchCFBase:
    """Lightweight base class implementing Torch-specific utilities."""

    similarity_metric: str = "cosine"
    k_neighbors: int = 50
    device: Optional[str] = None

    def __post_init__(self) -> None:
        self.device_obj = _resolve_device(self.device)
        supported = {"cosine", "pearson", "jaccard"}
        metric = (self.similarity_metric or "cosine").lower()
        if metric not in supported:
            raise ValueError(f"Torch backend currently supports {supported} similarity metrics")
        self.metric = metric
        self.is_fitted: bool = False
        self.similarity: Optional["torch.Tensor"] = None
        self.rating_matrix: Optional["torch.Tensor"] = None
        self.matrix_shape: Optional[Tuple[int, int]] = None
        self.similarity_time: float = 0.0
        self.user_item_matrix = None  # Retain reference for downstream tooling
        self.similarity_topk: Optional["torch.Tensor"] = None
        self.similarity_topk_abs: Optional["torch.Tensor"] = None
        self.topk_values: Optional["torch.Tensor"] = None
        self.topk_indices: Optional["torch.Tensor"] = None
        self.prediction_cache: Optional["torch.Tensor"] = None
        self.prediction_cache_np: Optional[np.ndarray] = None

    # --- helpers -----------------------------------------------------------------
    def _ensure_fitted(self) -> None:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before inference. Call fit() first.")

    def _similarity_cosine(self, matrix: "torch.Tensor") -> "torch.Tensor":
        normalized = torch.nn.functional.normalize(matrix, p=2, dim=1)
        return normalized @ normalized.T

    def _similarity_pearson(
        self,
        matrix: "torch.Tensor",
        means: "torch.Tensor",
        mask: "torch.Tensor",
    ) -> "torch.Tensor":
        centered = (matrix - means.unsqueeze(1)) * mask
        norms = torch.linalg.norm(centered, dim=1, keepdim=True) + 1e-12
        normalized = centered / norms
        return normalized @ normalized.T

    def _similarity_jaccard(self, mask: "torch.Tensor") -> "torch.Tensor":
        mask_float = mask.float()
        intersection = mask_float @ mask_float.T
        counts = mask.sum(dim=1, keepdim=True)
        unions = counts + counts.T - intersection
        unions = torch.clamp(unions, min=1.0)
        return intersection / unions

    def _select_similarity(
        self,
        matrix: "torch.Tensor",
        means: "torch.Tensor",
        mask: "torch.Tensor",
    ) -> "torch.Tensor":
        if self.metric == "cosine":
            return self._similarity_cosine(matrix)
        if self.metric == "pearson":
            return self._similarity_pearson(matrix, means, mask)
        if self.metric == "jaccard":
            return self._similarity_jaccard(mask)
        raise RuntimeError("Unsupported similarity metric")

    def _topk(
        self,
        sims: "torch.Tensor",
        mask: "torch.Tensor",
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        sims = sims * mask
        available = int(mask.sum().item())
        if available == 0:
            empty_values = torch.empty(0, device=self.device_obj)
            empty_indices = torch.empty(0, dtype=torch.long, device=self.device_obj)
            return empty_values, empty_indices
        k = min(self.k_neighbors, available)
        values, indices = torch.topk(sims, k)
        return values, indices

    def _synchronize(self) -> None:
        if self.device_obj.type == "cuda":
            torch.cuda.synchronize(self.device_obj)
        elif self.device_obj.type == "mps":
            # MPS synchronization for Apple Silicon
            if hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()

    def get_model_summary(self) -> Dict[str, Any]:
        """Expose summary info so downstream analytics can treat Torch models uniformly."""
        return {
            "model_type": self.__class__.__name__,
            "backend": "torch",
            "device": str(self.device_obj),
            "similarity_metric": self.metric,
            "k_neighbors": self.k_neighbors,
            "is_fitted": self.is_fitted,
            "matrix_shape": self.matrix_shape,
            "similarity_shape": list(self.similarity.shape) if self.similarity is not None else None,
            "similarity_computation_seconds": self.similarity_time,
            "prediction_cache_shape": list(self.prediction_cache.shape) if self.prediction_cache is not None else None,
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Provide a NumPy-backend compatible info dict (without backend tag)."""
        summary = self.get_model_summary().copy()
        summary.pop("backend", None)
        return summary


class TorchUserBasedCF(TorchCFBase):
    """User-based CF accelerated by PyTorch (CPU/GPU)."""

    def fit(self, user_item_matrix, timestamp_matrix: Optional[Any] = None) -> "TorchUserBasedCF":
        self.user_item_matrix = user_item_matrix
        ratings = _to_tensor(user_item_matrix, self.device_obj)
        self.rating_matrix = ratings
        self.matrix_shape = tuple(ratings.shape)
        self.user_means, self.user_mask, _ = _mean_ignore_zero(ratings, dim=1)
        global_sum = (ratings * self.user_mask).sum().item()
        global_count = self.user_mask.sum().item()
        self.global_mean = global_sum / global_count if global_count > 0 else 0.0

        sim_start = time.time()
        self.similarity = self._select_similarity(ratings, self.user_means, self.user_mask)
        self._synchronize()
        self.similarity_time = time.time() - sim_start
        self.similarity_matrix = self.similarity.detach().cpu().numpy()

        self.similarity_topk, self.similarity_topk_abs, self.topk_values, self.topk_indices = _build_sparse_topk(
            self.similarity.clone(),
            self.k_neighbors,
        )

        mask = self.user_mask
        rating_deviation = (ratings - self.user_means.unsqueeze(1)) * mask

        if self.similarity_topk._nnz() == 0:
            base_means = torch.where(
                self.user_means > 0,
                self.user_means,
                torch.full_like(self.user_means, self.global_mean),
            )
            prediction_gpu = base_means.unsqueeze(1).expand(-1, ratings.shape[1])
        else:
            numerator = torch.sparse.mm(self.similarity_topk, rating_deviation)
            denominator = torch.sparse.mm(self.similarity_topk_abs, mask)

            base_means = torch.where(
                self.user_means > 0,
                self.user_means,
                torch.full_like(self.user_means, self.global_mean),
            ).unsqueeze(1)

            denom_mask = denominator > 0
            denom_safe = torch.where(denom_mask, denominator, torch.ones_like(denominator))
            preds = base_means + numerator / denom_safe
            preds = torch.where(denom_mask, preds, base_means)
            prediction_gpu = preds

        min_rating, max_rating = cfg.model.rating_scale
        prediction_gpu = prediction_gpu.clamp(float(min_rating), float(max_rating))

        self.prediction_cache = prediction_gpu.detach().cpu()
        self.prediction_cache_np = self.prediction_cache.numpy()
        del prediction_gpu, rating_deviation

        self.is_fitted = True
        logger.info(
            f"[Torch] User-based similarity computed on {self.device_obj} in {self.similarity_time:.2f}s"
        )
        return self

    def predict(self, user_idx: int, item_idx: int, timestamp: Optional[float] = None) -> float:
        self._ensure_fitted()
        value = self.prediction_cache[user_idx, item_idx]
        return float(value.item())

    def predict_batch(self, user_item_pairs: Iterable[Tuple[int, int]]) -> np.ndarray:
        self._ensure_fitted()
        pairs = list(user_item_pairs)
        if not pairs:
            return np.array([], dtype=np.float32)
        users, items = zip(*pairs)
        preds = self.prediction_cache_np[users, items]
        return preds.astype(np.float32)

    def recommend(
        self,
        user_idx: int,
        n_recommendations: int = 10,
        exclude_rated: bool = True,
    ) -> List[Tuple[int, float]]:
        self._ensure_fitted()
        predictions = self.prediction_cache_np[user_idx]

        if exclude_rated and self.user_item_matrix is not None:
            rated = self.user_item_matrix[user_idx].toarray().ravel() > 0
        else:
            rated = np.zeros_like(predictions, dtype=bool)

        candidates = np.where(~rated)[0]
        if candidates.size == 0:
            return []

        top_n = min(n_recommendations, candidates.size)
        candidate_scores = predictions[candidates]
        top_indices = np.argpartition(-candidate_scores, top_n - 1)[:top_n]
        best_candidates = candidates[top_indices]
        best_scores = candidate_scores[top_indices]
        ordering = np.argsort(-best_scores)
        return [(int(best_candidates[idx]), float(best_scores[idx])) for idx in ordering]


class TorchItemBasedCF(TorchCFBase):
    """Item-based CF accelerated by PyTorch (CPU/GPU)."""

    def fit(self, user_item_matrix, timestamp_matrix: Optional[Any] = None) -> "TorchItemBasedCF":
        self.user_item_matrix = user_item_matrix
        ratings = _to_tensor(user_item_matrix, self.device_obj)
        self.rating_matrix = ratings
        self.matrix_shape = tuple(ratings.shape)

        item_matrix = ratings.T  # (items, users)
        self.item_means, self.item_mask, _ = _mean_ignore_zero(item_matrix, dim=1)
        global_sum = (ratings * (ratings > 0)).sum().item()
        global_count = (ratings > 0).sum().item()
        self.global_mean = global_sum / global_count if global_count > 0 else 0.0

        sim_start = time.time()
        self.similarity = self._select_similarity(item_matrix, self.item_means, self.item_mask)
        self._synchronize()
        self.similarity_time = time.time() - sim_start
        self.similarity_matrix = self.similarity.detach().cpu().numpy()

        self.similarity_topk, self.similarity_topk_abs, self.topk_values, self.topk_indices = _build_sparse_topk(
            self.similarity.clone(),
            self.k_neighbors,
        )

        mask = (ratings > 0).float()
        rating_deviation = (ratings - self.item_means.unsqueeze(0)) * mask

        if self.similarity_topk._nnz() == 0:
            base_means = torch.where(
                self.item_means > 0,
                self.item_means,
                torch.full_like(self.item_means, self.global_mean),
            )
            prediction_gpu = base_means.unsqueeze(0).expand(ratings.shape[0], -1)
        else:
            numerator_t = torch.sparse.mm(self.similarity_topk, rating_deviation.T)
            denominator_t = torch.sparse.mm(self.similarity_topk_abs, mask.T)
            numerator = numerator_t.T
            denominator = denominator_t.T

            base_means = torch.where(
                self.item_means > 0,
                self.item_means,
                torch.full_like(self.item_means, self.global_mean),
            ).unsqueeze(0)

            denom_mask = denominator > 0
            denom_safe = torch.where(denom_mask, denominator, torch.ones_like(denominator))
            preds = base_means + numerator / denom_safe
            preds = torch.where(denom_mask, preds, base_means)
            prediction_gpu = preds

        min_rating, max_rating = cfg.model.rating_scale
        prediction_gpu = prediction_gpu.clamp(float(min_rating), float(max_rating))

        self.prediction_cache = prediction_gpu.detach().cpu()
        self.prediction_cache_np = self.prediction_cache.numpy()
        del prediction_gpu, rating_deviation

        self.is_fitted = True
        logger.info(
            f"[Torch] Item-based similarity computed on {self.device_obj} in {self.similarity_time:.2f}s"
        )
        return self

    def predict(self, user_idx: int, item_idx: int, timestamp: Optional[float] = None) -> float:
        self._ensure_fitted()
        value = self.prediction_cache[user_idx, item_idx]
        return float(value.item())

    def predict_batch(self, user_item_pairs: Iterable[Tuple[int, int]]) -> np.ndarray:
        self._ensure_fitted()
        pairs = list(user_item_pairs)
        if not pairs:
            return np.array([], dtype=np.float32)
        users, items = zip(*pairs)
        preds = self.prediction_cache_np[users, items]
        return preds.astype(np.float32)

    def recommend(
        self,
        user_idx: int,
        n_recommendations: int = 10,
        exclude_rated: bool = True,
    ) -> List[Tuple[int, float]]:
        self._ensure_fitted()
        predictions = self.prediction_cache_np[user_idx]

        if exclude_rated and self.user_item_matrix is not None:
            rated = self.user_item_matrix[user_idx].toarray().ravel() > 0
        else:
            rated = np.zeros_like(predictions, dtype=bool)

        candidates = np.where(~rated)[0]
        if candidates.size == 0:
            return []

        top_n = min(n_recommendations, candidates.size)
        candidate_scores = predictions[candidates]
        top_indices = np.argpartition(-candidate_scores, top_n - 1)[:top_n]
        best_candidates = candidates[top_indices]
        best_scores = candidate_scores[top_indices]
        ordering = np.argsort(-best_scores)
        return [(int(best_candidates[idx]), float(best_scores[idx])) for idx in ordering]
