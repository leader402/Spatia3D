r"""ADMM deconvolution solver (C3).

Solves the OptiGraph3D MAP deconvolution as a regularised inverse problem. With proportions
``P`` (n_spots x n_celltypes), reference signatures ``V`` (n_celltypes x n_genes) and expression
``Y`` (n_spots x n_genes):

    min_P  1/2 ||Y - P V||_F^2  +  l1 ||P||_1  +  l2/2 ||P||_F^2  +  tv/2 tr(P^T L P)
    s.t.   P >= 0

i.e. Frobenius data fidelity + Elastic Net (l1 sparsity, l2 collinearity) + a quadratic spatial-TV
smoothness through the spot graph Laplacian ``L``, with non-negativity.

ADMM split ``P = Z``:
  f(P) = data + l2 + tv  (smooth quadratic),   g(Z) = l1||Z||_1 + indicator(Z >= 0).
The P-update is a Sylvester system ``A P + P B = C`` with ``A = tv L + (rho+l2) I`` (n x n) and
``B = V V^T`` (k x k). Diagonalising the *small* ``B = Q diag(w) Q^T`` decouples it into ``k``
independent SPD systems ``(tv L + (rho+l2+w_j) I) p_j = c_j`` — each solved directly (sparse
Cholesky) or by conjugate gradient (the route that scales to Stereo-seq spot counts). The Z-update
is a closed-form non-negative soft-threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.typing import ArrayLike

from spatia3d.deconvolution.graph import spatial_laplacian

__all__ = ["DeconvolutionResult", "deconvolve_admm"]


@dataclass
class DeconvolutionResult:
    """Output of :func:`deconvolve_admm`."""

    proportions: np.ndarray  # (n_spots, n_celltypes), row-normalised if normalize=True
    raw: np.ndarray  # solution before row normalisation
    n_iter: int
    primal_residual: float
    dual_residual: float
    converged: bool
    history: dict = field(default_factory=dict)
    proportions_sd: np.ndarray | None = None  # Laplace posterior SD if return_uncertainty=True


def _solve_pupdate(L, rho, l2, w, Ctil, solver, cg_tol, warm, extra_shift=0.0):
    """Solve the decoupled SPD systems for the diagonalised Sylvester P-update.

    ``extra_shift`` adds a diagonal term (e.g. the C1 domain-prior weight) to every system.
    """
    n, k = Ctil.shape
    shifts = rho + l2 + extra_shift + w  # (k,) per-column diagonal shift
    if L is None:  # no TV -> purely diagonal, fully separable across spots
        return Ctil / shifts[None, :]
    Ptil = np.empty_like(Ctil)
    Lsp = sp.csr_matrix(L)
    Ident = sp.identity(n, format="csr")
    for j in range(k):
        A_j = (Lsp + shifts[j] * Ident).tocsc()
        if solver == "cg":
            x0 = warm[:, j] if warm is not None else None
            Ptil[:, j], _ = spla.cg(A_j, Ctil[:, j], rtol=cg_tol, x0=x0, maxiter=500)
        else:
            Ptil[:, j] = spla.spsolve(A_j, Ctil[:, j])
    return Ptil


def deconvolve_admm(
    Y: ArrayLike,
    V: ArrayLike,
    *,
    coords: ArrayLike | None = None,
    laplacian: sp.spmatrix | None = None,
    k: int = 6,
    l1: float = 0.01,
    l2: float = 0.01,
    tv: float = 0.1,
    rho: float = 1.0,
    prior_target: ArrayLike | None = None,
    prior_weight: float = 0.0,
    max_iter: int = 300,
    tol: float = 1e-4,
    eps_rel: float = 1e-3,
    normalize: bool = True,
    solver: str = "direct",
    cg_tol: float = 1e-6,
    return_uncertainty: bool = False,
    uncertainty_method: str = "bootstrap",
    n_boot: int = 24,
    seed: int = 0,
    verbose: bool = False,
) -> DeconvolutionResult:
    """Deconvolve ``Y ≈ P V`` under Elastic Net + spatial TV + non-negativity via ADMM.

    Parameters
    ----------
    Y, V
        Expression ``(n_spots, n_genes)`` and reference signatures ``(n_celltypes, n_genes)``.
    coords, laplacian, k
        Spatial graph for the TV term. Provide a precomputed ``laplacian`` or ``coords`` (a k-NN
        Laplacian is then built). Required when ``tv > 0``.
    l1, l2, tv, rho
        Elastic-Net L1/L2 weights, TV weight, and ADMM penalty.
    max_iter, tol, eps_rel
        Iteration cap and the absolute / relative tolerances of the Boyd primal-dual stopping rule
        (tolerances scale with the variable magnitudes, so convergence is invariant to expression
        scale).
    normalize
        If True, row-normalise the solution to proportions that sum to one.
    solver, cg_tol
        ``"direct"`` (sparse Cholesky via spsolve) or ``"cg"`` (conjugate gradient, scalable).
    return_uncertainty, uncertainty_method, n_boot, seed
        If ``return_uncertainty``, also compute per-spot, per-cell-type SD into
        ``DeconvolutionResult.proportions_sd``. ``uncertainty_method`` is ``"bootstrap"`` (default;
        ``n_boot`` parametric refits, ``seed`` its RNG — discriminative but costs ``n_boot`` extra
        solves) or ``"laplace"`` (one-pass, free). See
        :func:`spatia3d.deconvolution.uncertainty.proportion_uncertainty`.
    """
    Y = np.asarray(Y, dtype=float)
    V = np.asarray(V, dtype=float)
    n, g = Y.shape
    kc = V.shape[0]
    if V.shape[1] != g:
        raise ValueError(f"gene dimension mismatch: Y has {g}, V has {V.shape[1]}")

    L = None
    if tv > 0:
        if laplacian is not None:
            L = laplacian
        elif coords is not None:
            L = spatial_laplacian(coords, k=k)
        else:
            raise ValueError("tv > 0 requires `coords` or `laplacian`")

    # Optional C1 domain-prior term  (prior_weight/2) ||P - prior_target||^2 (pulls each spot's
    # composition toward its domain mean); adds prior_weight to the P-update diagonal and to RHS.
    prior = None
    if prior_weight > 0:
        if prior_target is None:
            raise ValueError("prior_weight > 0 requires prior_target")
        prior = np.asarray(prior_target, dtype=float)
        if prior.shape != (n, kc):
            raise ValueError(f"prior_target shape {prior.shape} != ({n}, {kc})")

    # Diagonalise the small Gram matrix B = V V^T = Q diag(w) Q^T once.
    B = V @ V.T
    w, Q = np.linalg.eigh(B)
    YVt = Y @ V.T  # (n, kc)
    if prior is not None:
        YVt = YVt + prior_weight * prior  # constant RHS contribution of the prior term

    P = np.zeros((n, kc))
    Z = np.zeros((n, kc))
    U = np.zeros((n, kc))
    obj_hist, primal_hist, dual_hist = [], [], []
    primal = dual = np.inf
    converged = False
    n_iter = 0
    for _ in range(max_iter):
        n_iter += 1
        C = YVt + rho * (Z - U)
        Ctil = C @ Q
        Ptil = _solve_pupdate(
            L, rho, l2, w, Ctil, solver, cg_tol, warm=(Z @ Q), extra_shift=prior_weight
        )
        P = Ptil @ Q.T

        Z_prev = Z
        Z = np.maximum(P + U - l1 / rho, 0.0)  # prox of l1 + nonneg
        U = U + P - Z

        # Boyd et al. (2011) stopping rule: primal/dual residuals against scale-aware tolerances
        # (a single absolute tol is wrong here because P inherits the expression magnitude).
        primal = np.linalg.norm(P - Z)
        dual = rho * np.linalg.norm(Z - Z_prev)
        sqrt_n = np.sqrt(P.size)
        eps_pri = sqrt_n * tol + eps_rel * max(np.linalg.norm(P), np.linalg.norm(Z))
        eps_dual = sqrt_n * tol + eps_rel * rho * np.linalg.norm(U)
        primal_hist.append(primal)
        dual_hist.append(dual)
        if verbose or len(obj_hist) == 0:
            resid = 0.5 * np.linalg.norm(Y - P @ V) ** 2
            obj = (
                resid
                + l1 * np.abs(P).sum()
                + 0.5 * l2 * np.linalg.norm(P) ** 2
                + (0.5 * tv * np.sum(P * (L @ P)) if L is not None else 0.0)
            )
            obj_hist.append(float(obj))
        if primal < eps_pri and dual < eps_dual:
            converged = True
            break

    result_raw = Z  # Z carries the non-negativity / sparsity constraints exactly
    if normalize:
        row = result_raw.sum(axis=1, keepdims=True)
        proportions = np.divide(result_raw, row, out=np.zeros_like(result_raw), where=row > 0)
    else:
        proportions = result_raw

    proportions_sd = None
    if return_uncertainty:
        from spatia3d.deconvolution.uncertainty import proportion_uncertainty

        def _refit(Yb):
            # Re-solve with identical settings (no recursion: return_uncertainty defaults False).
            r = deconvolve_admm(
                Yb, V, laplacian=L, k=k, l1=l1, l2=l2, tv=tv, rho=rho,
                prior_target=prior_target, prior_weight=prior_weight, max_iter=max_iter,
                tol=tol, eps_rel=eps_rel, normalize=normalize, solver=solver, cg_tol=cg_tol,
            )
            return r.proportions if normalize else r.raw

        proportions_sd = proportion_uncertainty(
            Y, V, result_raw, method=uncertainty_method, normalize=normalize,
            n_boot=n_boot, refit=_refit, seed=seed,
        )

    return DeconvolutionResult(
        proportions=proportions,
        raw=result_raw,
        n_iter=n_iter,
        primal_residual=float(primal),
        dual_residual=float(dual),
        converged=converged,
        history={"objective": obj_hist, "primal": primal_hist, "dual": dual_hist},
        proportions_sd=proportions_sd,
    )
