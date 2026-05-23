"""Slice-alignment and 3D-continuity metrics.

These score how well two registered slices (or a stack) line up: paired-landmark error in physical
units, label-transfer accuracy across an interface, within-slice spatial coherence, and cross-slice
domain continuity. Coordinates are ``(n, d)`` arrays (d = 2 or 3).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.neighbors import NearestNeighbors

__all__ = [
    "landmark_error",
    "label_transfer_accuracy",
    "spatial_coherence_score",
    "cross_slice_label_continuity",
]


def landmark_error(coords_a: ArrayLike, coords_b: ArrayLike, *, aggregate: str = "mean") -> float:
    """Aggregated Euclidean distance between *paired* landmarks after alignment; lower is better.

    ``coords_a[i]`` and ``coords_b[i]`` are the same landmark in two slices. ``aggregate`` is one of
    ``"mean"``, ``"median"``, ``"rmse"``.
    """
    a = np.asarray(coords_a, dtype=float)
    b = np.asarray(coords_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"landmark shape mismatch: {a.shape} vs {b.shape}")
    dist = np.linalg.norm(a - b, axis=1)
    if aggregate == "mean":
        return float(dist.mean())
    if aggregate == "median":
        return float(np.median(dist))
    if aggregate == "rmse":
        return float(np.sqrt(np.mean(dist**2)))
    raise ValueError(f"unknown aggregate {aggregate!r}")


def _majority_label(neighbor_labels: np.ndarray) -> np.ndarray:
    """Row-wise majority vote over an (n, k) array of neighbour labels."""
    out = np.empty(neighbor_labels.shape[0], dtype=neighbor_labels.dtype)
    for i, row in enumerate(neighbor_labels):
        vals, counts = np.unique(row, return_counts=True)
        out[i] = vals[counts.argmax()]
    return out


def label_transfer_accuracy(
    coords_ref: ArrayLike,
    labels_ref: ArrayLike,
    coords_query: ArrayLike,
    labels_query: ArrayLike,
    *,
    k: int = 1,
) -> float:
    """Fraction of query spots whose ``k``-NN majority label in the reference slice matches.

    A well-aligned interface transfers labels accurately; higher is better (in [0, 1]).
    """
    ref = np.asarray(coords_ref, dtype=float)
    qry = np.asarray(coords_query, dtype=float)
    lref = np.asarray(labels_ref)
    lqry = np.asarray(labels_query)
    nn = NearestNeighbors(n_neighbors=min(k, ref.shape[0])).fit(ref)
    idx = nn.kneighbors(qry, return_distance=False)
    predicted = _majority_label(lref[idx])
    return float(np.mean(predicted == lqry))


def spatial_coherence_score(coords: ArrayLike, labels: ArrayLike, *, k: int = 6) -> float:
    """1 - PAS, where PAS is the fraction of spots disagreeing with their k-NN majority label.

    Measures within-slice spatial smoothness of a domain labelling; higher is better (in [0, 1]).
    """
    X = np.asarray(coords, dtype=float)
    y = np.asarray(labels)
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"coords/labels length mismatch: {X.shape[0]} vs {y.shape[0]}")
    # +1 neighbour because the first match is the point itself; drop that column.
    nn = NearestNeighbors(n_neighbors=min(k + 1, X.shape[0])).fit(X)
    idx = nn.kneighbors(X, return_distance=False)[:, 1:]
    neighbor_majority = _majority_label(y[idx])
    abnormal = np.mean(neighbor_majority != y)
    return float(1.0 - abnormal)


def cross_slice_label_continuity(
    coords_a: ArrayLike,
    labels_a: ArrayLike,
    coords_b: ArrayLike,
    labels_b: ArrayLike,
    *,
    k: int = 1,
) -> float:
    """Symmetric fraction of nearest cross-slice neighbours that share a domain label.

    Built on aligned in-plane coordinates of two adjacent slices; higher means the 3D stack has
    smoother domains across the interface (in [0, 1]).
    """
    a = np.asarray(coords_a, dtype=float)
    b = np.asarray(coords_b, dtype=float)
    la = np.asarray(labels_a)
    lb = np.asarray(labels_b)

    def _one_way(src, lsrc, dst, ldst):
        nn = NearestNeighbors(n_neighbors=min(k, dst.shape[0])).fit(dst)
        idx = nn.kneighbors(src, return_distance=False)
        matched = _majority_label(ldst[idx])
        return np.mean(matched == lsrc)

    return float(0.5 * (_one_way(a, la, b, lb) + _one_way(b, lb, a, la)))
