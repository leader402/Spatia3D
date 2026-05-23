"""Rigid coarse alignment: closed-form Procrustes (paired) and ICP (unpaired).

This is the *coarse-init* stage of C2 (PLAN.md). It plays the role NDT does in OptiGraph3D — a fast
rigid pre-alignment — but uses the standard, well-conditioned Kabsch/ICP estimators. The
differentiable non-rigid / OT refinement that makes alignment end-to-end (C2's novelty) is layered
on top once torch / POT are available.

Convention: a transform ``(R, t)`` maps ``source`` to ``target`` as ``source @ R.T + t``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.neighbors import NearestNeighbors

__all__ = ["kabsch", "apply_transform", "icp"]


def apply_transform(coords: ArrayLike, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Apply a rigid transform: ``coords @ R.T + t``."""
    return np.asarray(coords, dtype=float) @ R.T + t


def kabsch(
    source: ArrayLike, target: ArrayLike, *, allow_reflection: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Optimal rigid transform aligning *corresponding* points (Kabsch/orthogonal Procrustes).

    Returns ``(R, t)`` minimising ``||target - (source @ R.T + t)||`` over rotations. ``source`` and
    ``target`` must have matching rows (correspondence by index). With ``allow_reflection`` the
    estimate may be an improper rotation (det = -1).
    """
    P = np.asarray(source, dtype=float)
    Q = np.asarray(target, dtype=float)
    if P.shape != Q.shape:
        raise ValueError(f"source/target shape mismatch: {P.shape} vs {Q.shape}")
    cP, cQ = P.mean(0), Q.mean(0)
    M = (P - cP).T @ (Q - cQ)
    U, _, Vt = np.linalg.svd(M)
    V = Vt.T
    R = V @ U.T
    if not allow_reflection and np.linalg.det(R) < 0:
        V = V.copy()
        V[:, -1] *= -1
        R = V @ U.T
    t = cQ - R @ cP
    return R, t


def icp(
    source: ArrayLike,
    target: ArrayLike,
    *,
    max_iter: int = 50,
    tol: float = 1e-6,
    max_correspondence_dist: float | None = None,
    init: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Iterative Closest Point for *unpaired* point sets.

    Alternates nearest-neighbour matching with a Kabsch step until the mean correspondence distance
    stops improving. Returns ``(R, t, info)`` where ``info`` holds the iteration count and the final
    mean error. ``max_correspondence_dist`` optionally gates far matches (robustness to partial
    overlap); ``init`` provides a warm-start transform.
    """
    P = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    dim = P.shape[1]
    if init is None:
        # Centroid alignment removes the gross translation — without it ICP from identity stalls
        # in a local minimum whenever source and target are far apart.
        R_total = np.eye(dim)
        t_total = target.mean(0) - P.mean(0)
    else:
        R_total, t_total = init[0].copy(), init[1].copy()
    src = apply_transform(P, R_total, t_total)

    nn = NearestNeighbors(n_neighbors=1).fit(target)
    prev_err = np.inf
    info = {"iterations": 0, "mean_error": np.inf}
    for it in range(max_iter):
        dist, idx = nn.kneighbors(src)
        dist, idx = dist[:, 0], idx[:, 0]
        if max_correspondence_dist is not None:
            keep = dist < max_correspondence_dist
            if keep.sum() < dim + 1:  # too few correspondences to fit a transform
                break
            src_m, tgt_m = src[keep], target[idx[keep]]
            mean_err = float(dist[keep].mean())
        else:
            src_m, tgt_m = src, target[idx]
            mean_err = float(dist.mean())

        R_inc, t_inc = kabsch(src_m, tgt_m)
        R_total = R_inc @ R_total
        t_total = R_inc @ t_total + t_inc
        src = apply_transform(P, R_total, t_total)

        info = {"iterations": it + 1, "mean_error": mean_err}
        if abs(prev_err - mean_err) < tol:
            break
        prev_err = mean_err
    return R_total, t_total, info
