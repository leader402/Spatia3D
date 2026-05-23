"""Spatial-domain clustering metrics: ARI, NMI, ASW.

These quantify how well predicted spatial domains match a reference annotation (ARI/NMI) or how
well-separated they are in an embedding (ASW). All label functions can ignore unannotated spots,
matching real datasets such as DLPFC where a subset of spots carry no manual layer label.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

__all__ = ["ari", "nmi", "asw"]


def _valid_mask(labels: np.ndarray, ignore_label) -> np.ndarray:
    """Boolean mask of entries to keep (drop ``ignore_label`` and any NaN)."""
    mask = np.ones(labels.shape[0], dtype=bool)
    if ignore_label is not None:
        # `labels != nan` is always True, so NaN sentinels are handled below instead.
        is_nan_sentinel = isinstance(ignore_label, float) and np.isnan(ignore_label)
        if not is_nan_sentinel:
            mask &= labels != ignore_label
    if labels.dtype.kind in "fc":  # float/complex may contain NaN
        mask &= ~np.isnan(labels)
    return mask


def _aligned_labels(
    labels_true: ArrayLike, labels_pred: ArrayLike, ignore_label
) -> tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(labels_true)
    yp = np.asarray(labels_pred)
    if yt.shape[0] != yp.shape[0]:
        raise ValueError(f"label length mismatch: {yt.shape[0]} vs {yp.shape[0]}")
    mask = _valid_mask(yt, ignore_label) & _valid_mask(yp, ignore_label)
    if not mask.any():
        raise ValueError("no labelled spots remain after masking")
    return yt[mask], yp[mask]


def ari(labels_true: ArrayLike, labels_pred: ArrayLike, *, ignore_label=-1) -> float:
    """Adjusted Rand Index in [-1, 1]; 1.0 is a perfect match, ~0 is chance."""
    yt, yp = _aligned_labels(labels_true, labels_pred, ignore_label)
    return float(adjusted_rand_score(yt, yp))


def nmi(labels_true: ArrayLike, labels_pred: ArrayLike, *, ignore_label=-1) -> float:
    """Normalized Mutual Information in [0, 1]; 1.0 is a perfect match."""
    yt, yp = _aligned_labels(labels_true, labels_pred, ignore_label)
    return float(normalized_mutual_info_score(yt, yp))


def asw(
    embedding: ArrayLike,
    labels: ArrayLike,
    *,
    metric: str = "euclidean",
    scale: bool = False,
    ignore_label=-1,
) -> float:
    """Average Silhouette Width of ``labels`` in ``embedding`` space.

    Returns NaN when the silhouette is undefined (fewer than two clusters, or a cluster with a
    single member). With ``scale=True`` the score is mapped from [-1, 1] to [0, 1].
    """
    X = np.asarray(embedding, dtype=float)
    y = np.asarray(labels)
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"embedding/labels length mismatch: {X.shape[0]} vs {y.shape[0]}")
    mask = _valid_mask(y, ignore_label)
    X, y = X[mask], y[mask]
    if np.unique(y).size < 2 or np.unique(y).size >= X.shape[0]:
        return float("nan")
    score = float(silhouette_score(X, y, metric=metric))
    return (score + 1.0) / 2.0 if scale else score
