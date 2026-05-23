"""Cell-type deconvolution metrics: RMSE, JSD, PCC, AUPRC, rare-cell score.

Inputs are non-negative proportion matrices of shape ``(n_spots, n_celltypes)``: ``pred`` from a
method, ``true`` from a simulation ground truth. Rows are (approximately) compositions that sum to
one. These are the deconvolution-benchmark standards (Nat Commun 2023; Spotless eLife 2024).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import average_precision_score

__all__ = ["rmse", "jsd", "pcc", "auprc", "rare_cell_score", "deconvolution_report"]


def _check_pair(pred: ArrayLike, true: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(pred, dtype=float)
    t = np.asarray(true, dtype=float)
    if p.shape != t.shape:
        raise ValueError(f"shape mismatch: pred {p.shape} vs true {t.shape}")
    if p.ndim != 2:
        raise ValueError(f"expected 2-D (n_spots, n_celltypes), got ndim={p.ndim}")
    return p, t


def _safe_pcc(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r without scipy's constant-input warnings; NaN if either side is constant."""
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / denom) if denom > 0 else float("nan")


def rmse(pred: ArrayLike, true: ArrayLike, *, per_celltype: bool = False):
    """Root-mean-square error of proportions; lower is better.

    ``per_celltype=True`` returns one RMSE per cell type (column); otherwise a single scalar over
    all entries.
    """
    p, t = _check_pair(pred, true)
    if per_celltype:
        return np.sqrt(np.mean((p - t) ** 2, axis=0))
    return float(np.sqrt(np.mean((p - t) ** 2)))


def jsd(pred: ArrayLike, true: ArrayLike, *, eps: float = 1e-12) -> float:
    """Mean per-spot Jensen-Shannon divergence (base-2, in [0, 1]); lower is better.

    Each spot's rows are renormalised to a distribution before comparison, so the metric is robust
    to small departures from sum-to-one.
    """
    p, t = _check_pair(pred, true)
    p = (p + eps) / (p + eps).sum(axis=1, keepdims=True)
    t = (t + eps) / (t + eps).sum(axis=1, keepdims=True)
    m = 0.5 * (p + t)
    kl_pm = np.sum(p * np.log2(p / m), axis=1)
    kl_tm = np.sum(t * np.log2(t / m), axis=1)
    return float(np.mean(0.5 * kl_pm + 0.5 * kl_tm))


def pcc(pred: ArrayLike, true: ArrayLike, *, per_celltype: bool = True):
    """Pearson correlation coefficient; higher is better.

    ``per_celltype=True`` correlates predicted vs. true proportions across spots for each cell type
    and returns the mean over cell types (constant columns are skipped). Otherwise correlates the
    flattened matrices.
    """
    p, t = _check_pair(pred, true)
    if not per_celltype:
        return _safe_pcc(p.ravel(), t.ravel())
    per = np.array([_safe_pcc(p[:, k], t[:, k]) for k in range(p.shape[1])])
    if np.all(np.isnan(per)):
        return float("nan")
    return float(np.nanmean(per))


def auprc(pred: ArrayLike, true: ArrayLike, *, presence_threshold: float = 0.0) -> float:
    """Area under the precision-recall curve for cell-type *presence* detection; higher is better.

    A cell type is "present" in a spot when its true proportion exceeds ``presence_threshold``;
    predicted proportions act as the detection score. Returns NaN if presence is degenerate
    (all present or all absent).
    """
    p, t = _check_pair(pred, true)
    y_true = (t > presence_threshold).astype(int).ravel()
    if y_true.min() == y_true.max():
        return float("nan")
    return float(average_precision_score(y_true, p.ravel()))


def rare_cell_score(
    pred: ArrayLike,
    true: ArrayLike,
    *,
    rarity_threshold: float = 0.05,
    metric: str = "pcc",
):
    """Performance restricted to *rare* cell types (mean true proportion < ``rarity_threshold``).

    ``metric`` is ``"pcc"`` (higher better) or ``"rmse"`` (lower better), averaged over the rare
    cell types. Returns NaN if no cell type qualifies as rare.
    """
    p, t = _check_pair(pred, true)
    rare = np.where(t.mean(axis=0) < rarity_threshold)[0]
    if rare.size == 0:
        return float("nan")
    if metric == "pcc":
        vals = [_safe_pcc(p[:, k], t[:, k]) for k in rare]
        return float(np.nanmean(vals)) if not np.all(np.isnan(vals)) else float("nan")
    if metric == "rmse":
        return float(np.sqrt(np.mean((p[:, rare] - t[:, rare]) ** 2)))
    raise ValueError(f"unknown metric {metric!r}; use 'pcc' or 'rmse'")


def deconvolution_report(pred: ArrayLike, true: ArrayLike) -> dict[str, float]:
    """Convenience bundle of the standard deconvolution metrics as a flat dict."""
    return {
        "rmse": rmse(pred, true),
        "jsd": jsd(pred, true),
        "pcc": pcc(pred, true),
        "auprc": auprc(pred, true),
        "rare_pcc": rare_cell_score(pred, true, metric="pcc"),
    }
