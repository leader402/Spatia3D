"""Deconvolution method adapters for the benchmark.

In-env (no extra deps): NNLS, OLS-regression floor, and Spatia3D's ADMM solver. The deconvolution
gold standards (cell2location, RCTD, ...) pin conflicting dependency stacks, so they are wrapped as
isolated-env methods that raise MethodUnavailable until their env is built (docs/BENCHMARK.md).
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import nnls

from spatia3d.benchmark.base import DeconvolutionMethod, IsolatedEnvMethod

__all__ = ["NNLS", "OLSRegression", "SpatiaADMM", "Cell2location", "RCTD", "DECONVOLUTION_METHODS"]


def _row_normalize(P: np.ndarray) -> np.ndarray:
    s = P.sum(axis=1, keepdims=True)
    return np.divide(P, s, out=np.zeros_like(P), where=s > 0)


class NNLS(DeconvolutionMethod):
    """Per-spot non-negative least squares — the standard simple deconvolution baseline."""

    name = "NNLS"

    def run(self, Y: ArrayLike, V: ArrayLike, *, coords=None, **kw) -> np.ndarray:
        Y = np.asarray(Y, dtype=float)
        V = np.asarray(V, dtype=float)
        P = np.zeros((Y.shape[0], V.shape[0]))
        for i in range(Y.shape[0]):
            P[i], _ = nnls(V.T, Y[i])
        return _row_normalize(P)


class OLSRegression(DeconvolutionMethod):
    """Ordinary least squares with negatives clipped — the 'simple regression' floor that the
    Spotless benchmark (eLife 2024) showed beats half of the specialised methods."""

    name = "OLS-regression"

    def run(self, Y: ArrayLike, V: ArrayLike, *, coords=None, **kw) -> np.ndarray:
        Y = np.asarray(Y, dtype=float)
        V = np.asarray(V, dtype=float)
        coef, *_ = np.linalg.lstsq(V.T, Y.T, rcond=None)  # (n_celltypes, n_spots)
        return _row_normalize(np.clip(coef.T, 0, None))


class SpatiaADMM(DeconvolutionMethod):
    """Spatia3D's ADMM solver (Elastic Net + spatial-TV + non-negativity); uses the spot graph."""

    name = "Spatia3D-ADMM"
    is_ours = True

    def __init__(self, *, tv: float = 0.5, l1: float = 1e-3, l2: float = 1e-3, **kw) -> None:
        self.params = {"tv": tv, "l1": l1, "l2": l2, **kw}

    def run(
        self, Y: ArrayLike, V: ArrayLike, *, coords: ArrayLike | None = None, **kw
    ) -> np.ndarray:
        from spatia3d.deconvolution import deconvolve_admm

        tv = self.params.get("tv", 0.5)
        if tv > 0 and coords is None:
            raise ValueError("Spatia3D-ADMM with tv>0 needs coords for the spatial graph")
        res = deconvolve_admm(Y, V, coords=coords, **self.params)
        return res.proportions


class Cell2location(IsolatedEnvMethod, DeconvolutionMethod):
    """cell2location (Nat Biotech 2022) — Bayesian gold standard. Isolated env (scvi-tools)."""

    name = "cell2location"
    env_name = "spatia3d-c2l"

    def run(self, Y, V, *, coords=None, **kw):  # pragma: no cover - requires isolated env
        self._require_env()
        raise NotImplementedError("cell2location adapter: build env per docs/BENCHMARK.md")


class RCTD(IsolatedEnvMethod, DeconvolutionMethod):
    """RCTD / spacexr (Nat Biotech 2022) — probabilistic gold standard. Isolated R env."""

    name = "RCTD"
    env_name = "spatia3d-rctd"

    def run(self, Y, V, *, coords=None, **kw):  # pragma: no cover - requires isolated env
        self._require_env()
        raise NotImplementedError("RCTD R adapter: build env per docs/BENCHMARK.md")


def DECONVOLUTION_METHODS(*, include_ours: bool = True, include_isolated: bool = False):
    """Convenience factory: the default roster of deconvolution methods to benchmark."""
    methods: list[DeconvolutionMethod] = [NNLS(), OLSRegression()]
    if include_ours:
        methods.append(SpatiaADMM())
    if include_isolated:
        methods += [Cell2location(), RCTD()]
    return methods
