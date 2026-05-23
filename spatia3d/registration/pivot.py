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


def pivot_register(
    coords_list: list[ArrayLike],
    *,
    pivot: int | None = None,
    method: str = "paired",
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
            R, t, _ = icp(coords, target, **icp_kwargs)
        else:
            raise ValueError(f"unknown method {method!r}; use 'paired' or 'icp'")
        aligned.append(apply_transform(coords, R, t))
        transforms.append((R, t))
    return aligned, transforms
