"""Build the reference signature matrix ``V`` from an annotated single-cell dataset (C3 input step).

:func:`deconvolve_admm` takes a precomputed ``V`` ``(n_celltypes, n_genes)``; in practice that
matrix is *estimated* from an annotated scRNA-seq reference (counts + per-cell type labels) — the
same reference-profile step RCTD, CARD, and stereoscope run before deconvolving. This module is that
step: per-cell library-size normalisation (so deep and shallow cells contribute equally) followed by
the per-cell-type mean profile, with optional marker/HVG gene selection.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["build_signatures"]


def _normalize_counts(counts: np.ndarray, mode: str | None, target_sum: float | None) -> np.ndarray:
    """Per-cell library-size normalisation so each cell contributes at a common depth."""
    if mode is None:
        return counts
    if mode not in {"library", "cpm"}:
        raise ValueError(f"normalize must be 'library', 'cpm', or None, got {mode!r}")
    lib = counts.sum(axis=1, keepdims=True)
    target = (
        (1e4 if mode == "cpm" else float(np.median(lib[lib > 0])) if (lib > 0).any() else 1.0)
        if target_sum is None
        else float(target_sum)
    )
    return np.divide(counts, lib, out=np.zeros_like(counts), where=lib > 0) * target


def build_signatures(
    counts: ArrayLike,
    labels: ArrayLike,
    *,
    celltype_names: list | None = None,
    normalize: str | None = "library",
    target_sum: float | None = None,
    log1p: bool = False,
    n_markers: int | None = None,
) -> tuple[np.ndarray, list]:
    """Estimate the reference signature matrix ``V`` from an annotated scRNA reference.

    Parameters
    ----------
    counts
        ``(n_cells, n_genes)`` single-cell expression (raw counts or normalised).
    labels
        ``(n_cells,)`` cell-type label per cell (integer indices or strings).
    celltype_names
        Cell-type order for the rows of ``V``. Defaults to the sorted unique labels; every label
        must appear in it.
    normalize, target_sum
        Per-cell normalisation before averaging: ``"library"`` (scale each cell to ``target_sum``,
        default median library size), ``"cpm"`` (default target ``1e4``), or ``None`` (raw mean).
    log1p
        Apply ``log1p`` after normalisation (matches log-space deconvolution references).
    n_markers
        If given, keep only the union of each cell type's top-``n_markers`` genes (by mean profile),
        zeroing the rest — a light marker-gene focus. The returned ``V`` keeps all gene columns.

    Returns
    -------
    (V, celltype_names)
        ``V`` is ``(n_celltypes, n_genes)``; ``celltype_names`` is the row order.
    """
    counts = np.asarray(counts, dtype=float)
    labels = np.asarray(labels)
    if counts.ndim != 2:
        raise ValueError(f"counts must be 2-D (n_cells, n_genes), got ndim={counts.ndim}")
    if labels.shape[0] != counts.shape[0]:
        raise ValueError(
            f"labels ({labels.shape[0]}) and counts ({counts.shape[0]}) length mismatch"
        )

    if celltype_names is None:
        celltype_names = list(np.unique(labels))
    else:
        missing = set(np.unique(labels)) - set(celltype_names)
        if missing:
            raise ValueError(f"labels not in celltype_names: {sorted(missing)}")

    X = _normalize_counts(counts, normalize, target_sum)
    if log1p:
        X = np.log1p(X)

    V = np.zeros((len(celltype_names), counts.shape[1]))
    for r, name in enumerate(celltype_names):
        rows = labels == name
        if rows.any():
            V[r] = X[rows].mean(axis=0)

    if n_markers is not None and n_markers < counts.shape[1]:
        keep = np.zeros(counts.shape[1], dtype=bool)
        for r in range(V.shape[0]):
            keep[np.argsort(V[r])[::-1][:n_markers]] = True
        V[:, ~keep] = 0.0
    return V, list(celltype_names)
