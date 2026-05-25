"""Pivot-based stack registration.

Following OptiGraph3D, a single pivot slice (default: the middle one) is held fixed and every other
slice is aligned directly to it. Aligning all slices to one reference avoids the error accumulation
of sequential pairwise registration.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from spatia3d.registration.rigid import apply_transform, icp, kabsch

__all__ = ["pivot_register"]


def _inplane_rotation(theta: float, dim: int) -> np.ndarray:
    """In-plane rotation (about z for 3D) by ``theta``, as a ``(dim, dim)`` matrix."""
    c, s = np.cos(theta), np.sin(theta)
    R = np.eye(dim)
    R[:2, :2] = np.array([[c, -s], [s, c]])
    return R


def _icp_multi_angle(coords, target, n_angles, icp_kwargs):
    """ICP with ``n_angles`` evenly-spaced rotation inits, keeping the lowest-residual solve.

    init-free ICP/OT stalls on large inter-slice rotations (the reason PASTE struggles and STitch3D
    brute-forces angles); trying several starting rotations makes registration rotation-robust.
    """
    dim = coords.shape[1]
    best = None
    for kk in range(n_angles):
        R0 = _inplane_rotation(2 * np.pi * kk / n_angles, dim)
        t0 = target.mean(0) - coords.mean(0) @ R0.T  # centroid-align at this starting rotation
        R, t, info = icp(coords, target, init=(R0, t0), **icp_kwargs)
        if best is None or info["mean_error"] < best[2]:
            best = (R, t, info["mean_error"])
    return best[0], best[1]


def pivot_register(
    coords_list: list[ArrayLike],
    *,
    pivot: int | None = None,
    method: str = "paired",
    n_init_angles: int = 1,
    **icp_kwargs,
) -> tuple[list[np.ndarray], list[tuple[np.ndarray, np.ndarray]]]:
    """Align every slice to a pivot slice with a rigid transform.

    Parameters
    ----------
    coords_list
        One ``(n_i, d)`` coordinate array per slice.
    pivot
        Index of the fixed reference slice; defaults to the middle slice.
    method
        ``"paired"`` uses closed-form Kabsch and requires index-correspondence with the pivot (equal
        spot counts); ``"icp"`` uses ICP and works for unpaired sets.
    n_init_angles
        For ``method="icp"``: number of evenly-spaced rotation initialisations to try, keeping the
        best (lowest-residual) — makes ICP robust to large inter-slice rotation. ``1`` (default) is
        the plain centroid-init ICP; the field (STitch3D) brute-forces 6 angles for this reason.
    **icp_kwargs
        Forwarded to :func:`spatia3d.registration.rigid.icp` when ``method="icp"``.

    Returns
    -------
    aligned
        Coordinates of each slice mapped into the pivot frame.
    transforms
        The ``(R, t)`` applied to each slice (identity for the pivot).
    """
    if len(coords_list) == 0:
        raise ValueError("coords_list is empty")
    coords_list = [np.asarray(c, dtype=float) for c in coords_list]
    pivot = len(coords_list) // 2 if pivot is None else pivot
    target = coords_list[pivot]
    dim = target.shape[1]

    aligned: list[np.ndarray] = []
    transforms: list[tuple[np.ndarray, np.ndarray]] = []
    for i, coords in enumerate(coords_list):
        if i == pivot:
            R, t = np.eye(dim), np.zeros(dim)
        elif method == "paired":
            R, t = kabsch(coords, target)
        elif method == "icp":
            if n_init_angles > 1:
                R, t = _icp_multi_angle(coords, target, n_init_angles, icp_kwargs)
            else:
                R, t, _ = icp(coords, target, **icp_kwargs)
        else:
            raise ValueError(f"unknown method {method!r}; use 'paired' or 'icp'")
        aligned.append(apply_transform(coords, R, t))
        transforms.append((R, t))
    return aligned, transforms
