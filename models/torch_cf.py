"""PyTorch-based collaborative filtering models (experimental).

These classes mirror the interfaces of the NumPy-based implementations but run on
PyTorch tensors so that heavy similarity computations can be offloaded to GPU.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - torch may be optional
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False


def ensure_torch_available():
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for the 'torch' backend. Install torch first or use backend='numpy'."
        )


def _to_tensor(matrix, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(matrix):
        return matrix.to(device=device, dtype=torch.float32)
    if hasattr(matrix, "toarray"):
        dense = matrix.toarray().astype(np.float32, copy=False)
    else:
        dense = np.asarray(matrix, dtype=np.float32)
    return torch.from_numpy(dense).to(device)


def _normalize_rows(tensor: torch.Tensor) -> torch.Tensor:
    norms = torch.linalg.norm(tensor, dim=1, keepdim=True) + 1e-12
    return tensor / norms


@dataclass
class TorchUserBasedCF:
    similarity_metric: str = "cosine"
    k_neighbors: int = 50
    device: Optional[str] = None

    def __post_init__(self):
        ensure_torch_available()
        self.device_obj = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.similarity_metric != "cosine":
            raise ValueError("Torch backend currently supports only cosine similarity for user-based CF")

    def fit(self, user_item_matrix):
        ratings = _to_tensor(user_item_matrix, self.device_obj)
        self.rating_matrix = ratings
        normalized = _normalize_rows(ratings)
        start = time.time()
        self.similarity = normalized @ normalized.T
        if self.device_obj.type == "cuda":
            torch.cuda.synchronize(self.device_obj)
        self.similarity_time = time.time() - start

    def predict(self, user_idx: int, item_idx: int) -> float:
        sims = self.similarity[user_idx]
        ratings = self.rating_matrix[:, item_idx]
        mask = ratings > 0
        if mask.sum().item() == 0:
            return float("nan")
        sims = sims * mask
        topk = torch.topk(sims, min(self.k_neighbors, int(mask.sum().item())))
        neighbour_ratings = ratings[topk.indices]
        numerator = (topk.values * neighbour_ratings).sum()
        denom = topk.values.abs().sum()
        if denom.item() == 0:
            return float("nan")
        return (numerator / denom).item()


@dataclass
class TorchItemBasedCF:
    similarity_metric: str = "cosine"
    k_neighbors: int = 50
    device: Optional[str] = None

    def __post_init__(self):
        ensure_torch_available()
        self.device_obj = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.similarity_metric != "cosine":
            raise ValueError("Torch backend currently supports only cosine similarity for item-based CF")

    def fit(self, user_item_matrix):
        ratings = _to_tensor(user_item_matrix, self.device_obj)
        self.rating_matrix = ratings
        normalized = _normalize_rows(ratings.T)
        start = time.time()
        self.similarity = normalized @ normalized.T
        if self.device_obj.type == "cuda":
            torch.cuda.synchronize(self.device_obj)
        self.similarity_time = time.time() - start

    def predict(self, user_idx: int, item_idx: int) -> float:
        sims = self.similarity[item_idx]
        ratings = self.rating_matrix[user_idx]
        mask = ratings > 0
        if mask.sum().item() == 0:
            return float("nan")
        sims = sims * mask
        topk = torch.topk(sims, min(self.k_neighbors, int(mask.sum().item())))
        neighbour_ratings = ratings[topk.indices]
        numerator = (topk.values * neighbour_ratings).sum()
        denom = topk.values.abs().sum()
        if denom.item() == 0:
            return float("nan")
        return (numerator / denom).item()
