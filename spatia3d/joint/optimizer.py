"""C1 — unified joint deconvolution <-> domain optimizer (block-coordinate / unrolled form).

Formalises OptiGraph3D's "mutual supervision + convergence" idea. The two tasks share one coupled
energy (β = proportions, z/μ = domain labels/centroids, L = spatial graph Laplacian):

    F(β, z, μ) = ½‖Y − βV‖²_F + l1‖β‖₁ + l2/2‖β‖²_F + tv/2·tr(βᵀLβ)
               + κ/2·‖β − μ[z]‖²_F            (domain-conditioned composition prior)   s.t. β ≥ 0

We minimise F by alternating two blocks, each of which decreases F, so F is monotone non-increasing
and the scheme converges (block-coordinate descent; the domain block is exactly k-means on β):

  * domain block : (z, μ) ← argmin_{z,μ} ‖β − μ[z]‖²   (KMeans on β)            → z informs β
  * deconv block : β ← argmin_β F(·, z, μ)             (ADMM with prior μ[z])   → β informs z

So deconvolution and domain identification act as priors for each other and are solved jointly,
rather than the one-directional pipeline of prior work. This is the headline contribution (C1).
A GPU, end-to-end-differentiable variant (unrolled ADMM + soft k-means) lives in ``unrolled.py``
with RMSE parity to this exact version; SE(3)-registration coupling layers on top.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike
from sklearn.cluster import KMeans

from spatia3d.deconvolution import deconvolve_admm, spatial_laplacian

__all__ = ["JointResult", "joint_deconvolution_domain"]


@dataclass
class JointResult:
    proportions: np.ndarray  # (n_spots, n_celltypes) row-normalised
    domains: np.ndarray  # (n_spots,) domain labels
    centroids: np.ndarray  # (n_domains, n_celltypes) domain mean compositions
    energy: list[float]  # coupled energy F per outer iteration (monotone non-increasing)
    n_iter: int
    converged: bool
    raw: np.ndarray = field(repr=False, default=None)  # unnormalised β


def _energy(Y, V, beta, L, l1, l2, tv, kappa, B):
    e = 0.5 * np.linalg.norm(Y - beta @ V) ** 2
    e += l1 * np.abs(beta).sum() + 0.5 * l2 * np.linalg.norm(beta) ** 2
    if L is not None:
        e += 0.5 * tv * np.sum(beta * (L @ beta))
    if B is not None:
        e += 0.5 * kappa * np.linalg.norm(beta - B) ** 2
    return float(e)


def joint_deconvolution_domain(
    Y: ArrayLike,
    V: ArrayLike,
    *,
    n_domains: int,
    coords: ArrayLike | None = None,
    laplacian=None,
    k: int = 6,
    prior_weight: float = 1.0,
    tv: float = 0.5,
    l1: float = 1e-3,
    l2: float = 1e-3,
    n_outer: int = 15,
    tol: float = 1e-3,
    admm_max_iter: int = 300,
    seed: int = 0,
    verbose: bool = False,
) -> JointResult:
    """Jointly deconvolve and identify spatial domains by minimising the coupled energy F.

    Returns proportions, domain labels, the per-iteration energy (monotone non-increasing — the
    convergence certificate), and convergence info. ``prior_weight`` (κ) controls how strongly the
    domain composition prior denoises β; ``n_outer`` caps outer block-coordinate sweeps.
    """
    Y = np.asarray(Y, dtype=float)
    V = np.asarray(V, dtype=float)
    L = None
    if tv > 0:
        L = laplacian if laplacian is not None else spatial_laplacian(coords, k=k)

    def _deconv(prior):
        kw = {} if prior is None else {"prior_target": prior, "prior_weight": prior_weight}
        return deconvolve_admm(
            Y, V, laplacian=L, tv=tv, l1=l1, l2=l2, max_iter=admm_max_iter, normalize=False, **kw
        ).raw

    beta = _deconv(None)  # init: deconvolution with no domain prior
    energy: list[float] = []
    converged = False
    n_iter = 0
    for _ in range(n_outer):
        n_iter += 1
        km = KMeans(n_clusters=n_domains, n_init=10, random_state=seed).fit(beta)
        B = km.cluster_centers_[km.labels_]  # domain-conditioned prior target μ[z]
        beta_new = _deconv(B)
        energy.append(_energy(Y, V, beta_new, L, l1, l2, tv, prior_weight, B))
        delta = np.linalg.norm(beta_new - beta) / (np.linalg.norm(beta) + 1e-12)
        beta = beta_new
        if verbose:
            print(f"outer {n_iter:2d}  energy {energy[-1]:.4g}  Δβ {delta:.2e}")
        if delta < tol:
            converged = True
            break

    final = KMeans(n_clusters=n_domains, n_init=10, random_state=seed).fit(beta)
    row = beta.sum(axis=1, keepdims=True)
    proportions = np.divide(beta, row, out=np.zeros_like(beta), where=row > 0)
    return JointResult(
        proportions=proportions,
        domains=final.labels_,
        centroids=final.cluster_centers_,
        energy=energy,
        n_iter=n_iter,
        converged=converged,
        raw=beta,
    )
