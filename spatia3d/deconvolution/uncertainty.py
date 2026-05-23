r"""Posterior uncertainty for ADMM deconvolution (C3).

A point estimate alone is a known weakness: the gold-standard competitors quantify uncertainty
(RCTD reports per-spot confidence, cell2location a full variational posterior), and reviewers expect
a deconvolution method to say *how sure* it is. Two complementary estimators are provided:

* ``"laplace"`` — observed Fisher information of the Gaussian data term. The abundance covariance is
  ``Sigma = sigma² (V Vᵀ + ridge·I)⁻¹`` (same for every spot, as in homoscedastic least squares),
  propagated to proportions through the simplex map ``p ↦ p/Σp`` by the delta method. One ``(k, k)``
  eigendecomposition for the whole dataset — essentially free — well calibrated *in aggregate* and
  noise-responsive, but per-spot variation enters only through the Jacobian, so it is weakly
  discriminative (it says how uncertain the dataset is, not which spots are worst).

* ``"bootstrap"`` — parametric residual bootstrap. Resample ``Y_b ~ N(P̂ V, sigma²)`` and re-solve
  the deconvolution ``n_boot`` times; the per-entry SD across refits is the sampling spread of the
  actual (regularised, thresholded) estimator. Noise-responsive **and** discriminative — it flags
  the spots and cell types whose estimates are least reliable — at the cost of ``n_boot`` solves.
  This is the default for trustworthy per-spot uncertainty; drop to ``"laplace"`` (or lower
  ``n_boot``) on million-spot data.

Both estimate sampling *variance*, not the shrinkage *bias* the Elastic-Net / non-negativity impose
on the MAP estimate, so absolute interval coverage runs a little below nominal — the standard
behaviour of a regularised estimator (validated by :func:`spatia3d.metrics.credible_coverage`).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["proportion_uncertainty"]


def _residual_sigma2(Y: np.ndarray, V: np.ndarray, P: np.ndarray, eps: float) -> float:
    """Global homoscedastic noise variance estimate ``||Y - P V||²_F / (n_spots · n_genes)``."""
    resid = Y - P @ V
    return max(float((resid**2).mean()), eps)


def _simplex_delta_sd(cov_eig: np.ndarray, Q: np.ndarray, P: np.ndarray, eps: float) -> np.ndarray:
    """Per-spot SD on proportions: push abundance covariance through ``p ↦ p/Σp`` (delta method).

    ``cov_eig`` holds, per spot ``(n, k)``, the eigenvalues of that spot's abundance covariance in
    the shared eigenbasis ``Q`` (``Sigma_i = Q diag(cov_eig_i) Qᵀ``); a single row (constant
    covariance across spots) is accepted and reused for every spot.
    """
    n, k = P.shape
    sd = np.zeros((n, k))
    ones = np.ones(k)
    eye = np.eye(k)
    single = cov_eig.shape[0] == 1
    for i in range(n):
        s = P[i].sum()
        if s <= eps:
            continue  # an all-zero spot carries no proportion mass to be uncertain about
        cov = (Q * cov_eig[0 if single else i]) @ Q.T  # Q diag Qᵀ, no explicit inverse
        f = P[i] / s
        J = (eye - np.outer(f, ones)) / s  # J[a,b] = (δ_ab − f_a)/s
        sd[i] = np.sqrt(np.clip(np.diag(J @ cov @ J.T), 0.0, None))
    return sd


def _laplace_sd(
    Y: np.ndarray, V: np.ndarray, P: np.ndarray, sigma2: float, normalize: bool, ridge: float,
    eps: float,
) -> np.ndarray:
    """Observed-Fisher SD: ``Sigma = sigma² (V Vᵀ + ridge I)⁻¹``, optional simplex delta method."""
    g_eig, Q = np.linalg.eigh(V @ V.T)
    cov_eig = sigma2 / (np.clip(g_eig, 0.0, None) + ridge)  # (k,) eigenvalues of Sigma
    if not normalize:
        var = (Q**2) @ cov_eig  # diag(Sigma)_c = Σ_j Q[c,j]² cov_eig_j — same for every spot
        return np.broadcast_to(np.sqrt(np.clip(var, 0.0, None)), P.shape).copy()
    return _simplex_delta_sd(cov_eig[None, :], Q, P, eps)


def _bootstrap_sd(
    Y: np.ndarray, V: np.ndarray, P: np.ndarray, refit: Callable[[np.ndarray], ArrayLike],
    sigma2: float, n_boot: int, seed: int,
) -> np.ndarray:
    """Parametric residual bootstrap: SD of the refit estimator over ``Y_b ~ N(P̂ V, sigma²)``."""
    fit = P @ V
    sigma = float(np.sqrt(sigma2))
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_boot):
        Yb = np.clip(fit + rng.normal(0.0, sigma, size=Y.shape), 0.0, None)
        samples.append(np.asarray(refit(Yb), dtype=float))
    return np.std(np.stack(samples, axis=0), axis=0)


def proportion_uncertainty(
    Y: ArrayLike,
    V: ArrayLike,
    P_raw: ArrayLike,
    *,
    method: str = "bootstrap",
    sigma2: float | None = None,
    normalize: bool = True,
    ridge: float = 1e-6,
    n_boot: int = 24,
    refit: Callable[[np.ndarray], ArrayLike] | None = None,
    seed: int = 0,
    eps: float = 1e-12,
) -> np.ndarray:
    """Per-spot, per-cell-type uncertainty (SD) of a deconvolution fit.

    Parameters
    ----------
    Y, V
        Expression ``(n_spots, n_genes)`` and reference signatures ``(n_celltypes, n_genes)`` — the
        same arrays passed to :func:`deconvolve_admm`.
    P_raw
        The fitted *raw* (pre-normalisation) abundances ``(n_spots, n_celltypes)``
        (``DeconvolutionResult.raw``).
    method
        ``"bootstrap"`` (default; discriminative per-spot uncertainty, needs ``refit``) or
        ``"laplace"`` (one-pass observed-Fisher SD, no refit). See the module docstring.
    sigma2
        Observation-noise variance; defaults to the global residual variance of the fit.
    normalize
        If True (default), SD on the row-normalised proportions (delta method through the simplex);
        otherwise SD on the raw abundances.
    ridge
        Numerical ridge on the data curvature for ``method="laplace"`` (stabilises ``V Vᵀ``).
    n_boot, refit, seed
        For ``method="bootstrap"``: number of resamples, the callable ``Y_b -> proportions`` that
        re-solves the deconvolution (its output must match ``normalize``), and the RNG seed.
        :func:`deconvolve_admm` supplies ``refit`` automatically when ``return_uncertainty=True``.

    Returns
    -------
    np.ndarray
        ``(n_spots, n_celltypes)`` standard deviations, aligned with the proportion matrix.
    """
    Y = np.asarray(Y, dtype=float)
    V = np.asarray(V, dtype=float)
    P = np.asarray(P_raw, dtype=float)
    n, g = Y.shape
    k = V.shape[0]
    if P.shape != (n, k):
        raise ValueError(f"P_raw shape {P.shape} != ({n}, {k})")
    if sigma2 is None:
        sigma2 = _residual_sigma2(Y, V, P, eps)
    sigma2 = max(sigma2, eps)

    if method == "laplace":
        return _laplace_sd(Y, V, P, sigma2, normalize, ridge, eps)
    if method == "bootstrap":
        if refit is None:
            raise ValueError(
                "method='bootstrap' needs a `refit` callable (Y_b -> proportions); "
                "use deconvolve_admm(..., return_uncertainty=True) to wire it automatically, "
                "or pass method='laplace' for the one-pass estimate."
            )
        return _bootstrap_sd(Y, V, P, refit, sigma2, n_boot, seed)
    raise ValueError(f"unknown method {method!r}; use 'bootstrap' or 'laplace'")
