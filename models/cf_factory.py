"""Factory helpers to instantiate collaborative filtering models."""
from __future__ import annotations

from typing import Dict

from models.user_based_cf import UserBasedCollaborativeFiltering
from models.item_based_cf import ItemBasedCollaborativeFiltering
from models.temporal_cf import (
    TemporalUserBasedCollaborativeFiltering,
    TemporalItemBasedCollaborativeFiltering,
)

try:
    from models.torch_cf import TorchUserBasedCF, TorchItemBasedCF, ensure_torch_available
except ImportError:  # pragma: no cover
    TorchUserBasedCF = TorchItemBasedCF = None  # type: ignore
    def ensure_torch_available():  # type: ignore
        raise ImportError("PyTorch backend requested but torch is not installed.")


def build_cf_model(model_type: str, params: Dict, backend: str = "numpy", device: str = "cpu"):
    backend = (backend or "numpy").lower()
    model_type = model_type.lower()

    if backend == "torch":
        ensure_torch_available()
        if model_type == 'user_cf':
            return TorchUserBasedCF(
                similarity_metric=params.get('similarity_metric', 'cosine'),
                k_neighbors=params.get('k_neighbors', 50),
                device=device
            )
        elif model_type == 'item_cf':
            return TorchItemBasedCF(
                similarity_metric=params.get('similarity_metric', 'cosine'),
                k_neighbors=params.get('k_neighbors', 50),
                device=device
            )
        elif model_type in {'temporal_user_cf', 'temporal_item_cf'}:
            raise ValueError("Temporal collaborative filtering is not available for the PyTorch backend yet.")
        else:
            raise ValueError(f"Unknown model type {model_type}")

    # Default NumPy/scikit implementation
    if model_type == 'user_cf':
        return UserBasedCollaborativeFiltering(
            similarity_metric=params.get('similarity_metric', 'cosine'),
            k_neighbors=params.get('k_neighbors', 50)
        )
    elif model_type == 'item_cf':
        return ItemBasedCollaborativeFiltering(
            similarity_metric=params.get('similarity_metric', 'cosine'),
            k_neighbors=params.get('k_neighbors', 50)
        )
    elif model_type == 'temporal_user_cf':
        return TemporalUserBasedCollaborativeFiltering(
            similarity_metric=params.get('similarity_metric', 'cosine'),
            k_neighbors=params.get('k_neighbors', 50),
            half_life=params.get('half_life'),
            decay_strategy=params.get('decay_strategy'),
            decay_floor=params.get('decay_floor')
        )
    elif model_type == 'temporal_item_cf':
        return TemporalItemBasedCollaborativeFiltering(
            similarity_metric=params.get('similarity_metric', 'cosine'),
            k_neighbors=params.get('k_neighbors', 50),
            half_life=params.get('half_life'),
            decay_strategy=params.get('decay_strategy'),
            decay_floor=params.get('decay_floor')
        )
    else:
        raise ValueError(f"Unknown model type {model_type}")
