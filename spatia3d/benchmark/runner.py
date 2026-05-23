"""Benchmark runners: apply a roster of methods to a dataset, score against ground truth, tabulate.

Each runner returns a list of per-method result dicts (and a pandas DataFrame if pandas is present).
Methods that raise :class:`MethodUnavailable` are recorded as skipped rather than crashing the run,
so a benchmark proceeds with whatever subset of competitors is installed.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from spatia3d.benchmark.base import (
    AlignmentMethod,
    DeconvolutionMethod,
    DomainMethod,
    MethodUnavailable,
)
from spatia3d.metrics import (
    ari,
    deconvolution_report,
    landmark_error,
    nmi,
    spatial_coherence_score,
)
from spatia3d.utils import Timer

__all__ = [
    "benchmark_deconvolution",
    "benchmark_alignment",
    "benchmark_domain",
    "to_dataframe",
]


def to_dataframe(rows: list[dict]):
    """Convert result rows to a pandas DataFrame indexed by method (if pandas is available)."""
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover
        return rows
    return pd.DataFrame(rows).set_index("method")


def _run_one(method, fn) -> dict:
    """Run ``fn`` (which calls ``method.run``) with timing + graceful skip; return a result row."""
    row: dict = {"method": method.name, "ours": method.is_ours}
    try:
        with Timer() as t:
            metrics = fn()
        row.update(metrics)
        row["runtime_s"] = round(t.elapsed, 4)
        row["status"] = "ok"
    except MethodUnavailable as exc:
        row["status"] = f"skipped: {exc}"
    except Exception as exc:  # one method failing must not crash the whole benchmark
        row["status"] = f"error: {type(exc).__name__}: {str(exc)[:80]}"
    return row


def benchmark_deconvolution(
    methods: list[DeconvolutionMethod],
    Y: ArrayLike,
    V: ArrayLike,
    true_proportions: ArrayLike,
    *,
    coords: ArrayLike | None = None,
) -> list[dict]:
    """Score deconvolution methods (RMSE/JSD/PCC/AUPRC/rare-cell) vs. ground-truth proportions."""

    def _score(m):
        return deconvolution_report(m.run(Y, V, coords=coords), true_proportions)

    return [_run_one(m, lambda m=m: _score(m)) for m in methods]


def _pairwise_consistency(aligned: list[np.ndarray]) -> float:
    """Mean inter-slice landmark disagreement (frame-invariant): corresponding (same-index) points
    on adjacent slices should coincide after a good alignment, regardless of the global frame."""
    errs = [
        landmark_error(aligned[i], aligned[i + 1]) for i in range(len(aligned) - 1)
    ]
    return float(np.mean(errs))


def _error_vs_truth(aligned: list[np.ndarray], truth: list[np.ndarray]) -> float:
    """Landmark error to the canonical geometry after removing the global frame via one Kabsch fit
    over all points (so methods anchored in different frames are compared fairly)."""
    from spatia3d.registration import apply_transform, kabsch

    A = np.vstack([np.asarray(a, float) for a in aligned])
    T = np.vstack([np.asarray(t, float) for t in truth])
    R, t = kabsch(A, T)
    return float(landmark_error(apply_transform(A, R, t), T))


def benchmark_alignment(
    methods: list[AlignmentMethod],
    coords_list: list[ArrayLike],
    true_coords_list: list[ArrayLike],
    *,
    features: list[ArrayLike] | None = None,
) -> list[dict]:
    """Score alignment: inter-slice consistency + error vs. truth (post global Procrustes)."""
    truth = [np.asarray(c, float) for c in true_coords_list]

    def _score(m):
        aligned = m.run(coords_list, features=features)
        return {
            "consistency": _pairwise_consistency(aligned),
            "error_vs_truth": _error_vs_truth(aligned, truth),
        }

    return [_run_one(m, lambda m=m: _score(m)) for m in methods]


def benchmark_domain(
    methods: list[DomainMethod],
    X: ArrayLike,
    coords: ArrayLike,
    labels_true: ArrayLike,
    *,
    n_clusters: int,
) -> list[dict]:
    """Score domain methods (ARI/NMI + within-slice spatial coherence) against manual annotation."""
    rows = []

    def _score(m):
        pred = m.run(X, coords, n_clusters=n_clusters)
        return {
            "ARI": ari(labels_true, pred, ignore_label=None),
            "NMI": nmi(labels_true, pred, ignore_label=None),
            "coherence": spatial_coherence_score(coords, pred),
        }

    for m in methods:
        rows.append(_run_one(m, lambda m=m: _score(m)))
    return rows
