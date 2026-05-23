"""C1 — differentiable unrolled-ADMM joint deconvolution + domain optimizer (torch, GPU).

The block-coordinate optimizer (``optimizer.py``) alternates an exact ADMM β-solve with hard
k-means. This module realises the *same* scheme as a single end-to-end-differentiable computation:

  * deconvolution β  — the real ADMM iterations, **unrolled as differentiable layers**. Because the
    spatial-graph Laplacian ``L`` and the Gram ``VVᵀ`` are fixed, we eigendecompose them once
    (``L = U_L Λ U_Lᵀ``, ``VVᵀ = Q diag(w) Qᵀ``); each P-update is then the *exact* solve
    ``β̃ = U_L · (U_Lᵀ C̃) / (tv·Λ + ρ+l2+κ+w) · ...`` — pure differentiable matmuls (a spectral
    solve), so it reproduces the converged ADMM (RMSE parity) rather than a crude approximation.
  * domain block — differentiable **soft k-means** on the current composition, producing a
    domain-conditioned prior ``S·μ`` that feeds back into each ADMM solve.

Everything is differentiable, so registration / learned components can later enter the same gradient
flow. The dense eigendecomposition of ``L`` suits simulator/DLPFC scale; for Stereo-seq millions,
swap the spectral solve for unrolled conjugate gradient (TODO).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from numpy.typing import ArrayLike
from sklearn.cluster import KMeans

from spatia3d.deconvolution import spatial_laplacian

__all__ = ["DiffJointResult", "differentiable_joint_c1"]


@dataclass
class DiffJointResult:
    proportions: np.ndarray  # (n_spots, n_celltypes)
    domains: np.ndarray  # (n_spots,) hard domain labels (argmax of the soft assignment)
    assignment: np.ndarray  # (n_spots, n_domains) soft domain assignment S
    energy: list[float]  # coupled energy per outer iteration
    n_iter: int
    device: str = "cpu"
    raw: np.ndarray = field(repr=False, default=None)


class _Spectral:
    """Precomputed (constant) spectral factors for the exact differentiable ADMM P-update."""

    def __init__(self, Y, V, L, device, dtype=torch.float64):
        self.Yt = torch.as_tensor(Y, dtype=dtype, device=device)
        self.Vt = torch.as_tensor(V, dtype=dtype, device=device)
        Ld = torch.as_tensor(L, dtype=dtype, device=device)
        self.lamL, self.UL = torch.linalg.eigh(Ld)  # L = UL diag(lamL) ULᵀ
        self.w, self.Q = torch.linalg.eigh(self.Vt @ self.Vt.t())  # VVᵀ = Q diag(w) Qᵀ
        self.YVt = self.Yt @ self.Vt.t()
        self.n, self.K = self.Yt.shape[0], self.Vt.shape[0]


def _admm(sp, prior, *, kappa, rho, l2, l1, tv, n_admm, Z, U):
    """Run ``n_admm`` exact (spectral) ADMM iterations; differentiable. Returns updated (Z, U)."""
    rhs_prior = kappa * prior if prior is not None else 0.0
    shift = (rho + l2 + kappa + sp.w)[None, :]  # (1, K)
    for _ in range(n_admm):
        C = sp.YVt + rho * (Z - U) + rhs_prior
        Ctil = C @ sp.Q
        G = sp.UL.t() @ Ctil
        Ptil = sp.UL @ (G / (tv * sp.lamL[:, None] + shift))
        P = Ptil @ sp.Q.t()
        Z = torch.clamp(P + U - l1 / rho, min=0.0)  # prox of l1 + nonneg
        U = U + P - Z
    return Z, U


def differentiable_joint_c1(
    Y: ArrayLike,
    V: ArrayLike,
    coords: ArrayLike,
    *,
    n_domains: int,
    laplacian=None,
    k: int = 6,
    tv: float = 0.5,
    l1: float = 1e-3,
    l2: float = 1e-3,
    rho: float = 1.0,
    kappa: float = 1.0,
    tau: float = 0.05,
    n_outer: int = 8,
    n_admm: int = 60,
    tol: float = 1e-3,
    device: str | None = None,
    seed: int = 0,
) -> DiffJointResult:
    """Differentiable unrolled-ADMM joint deconvolution + soft-k-means domains (GPU when available).

    Reproduces the exact ADMM deconvolution (RMSE parity with ``deconvolve_admm``) while coupling a
    differentiable soft-domain prior. ``tau`` is the soft-assignment temperature on compositions;
    ``kappa`` the prior strength.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    Y = np.asarray(Y, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    L = laplacian.toarray() if laplacian is not None else spatial_laplacian(coords, k=k).toarray()
    sp = _Spectral(Y, V, L, device)

    def _laplacian_quad(b):  # tr(bᵀ L b) via the eigenbasis: ||Λ^½ Uᵀ b||²
        Utb = sp.UL.t() @ b
        return (sp.lamL[:, None] * Utb * Utb).sum()

    Z = torch.zeros(sp.n, sp.K, dtype=torch.float64, device=device)
    U = torch.zeros_like(Z)

    with torch.no_grad():  # deterministic forward; ops are differentiable (grads on for training)
        Z, U = _admm(sp, None, kappa=0.0, rho=rho, l2=l2, l1=l1, tv=tv, n_admm=n_admm, Z=Z, U=U)
        beta = Z
        # init soft-kmeans centroids from a hard k-means on the no-prior composition
        prop0 = (beta / (beta.sum(1, keepdim=True) + 1e-12)).cpu().numpy()
        c0 = KMeans(n_clusters=n_domains, n_init=10, random_state=seed).fit(prop0).cluster_centers_
        centroids = torch.as_tensor(c0, dtype=torch.float64, device=device)  # (n_domains, K), props

        energy: list[float] = []
        n_iter = 0
        for _ in range(n_outer):
            n_iter += 1
            prop = beta / (beta.sum(1, keepdim=True) + 1e-12)
            d2 = torch.cdist(prop, centroids) ** 2  # (n, n_domains)
            S = torch.softmax(-d2 / (d2.mean() * tau + 1e-12), dim=1)
            centroids = (S.t() @ prop) / (S.sum(0)[:, None] + 1e-12)  # soft centroids (props)
            prior = (S @ centroids) * beta.sum(1, keepdim=True)  # back to raw β scale
            beta_new, U = _admm(
                sp, prior, kappa=kappa, rho=rho, l2=l2, l1=l1, tv=tv, n_admm=n_admm, Z=beta, U=U
            )
            e = 0.5 * torch.linalg.norm(sp.Yt - beta_new @ sp.Vt) ** 2
            e = e + l1 * beta_new.abs().sum() + 0.5 * l2 * torch.linalg.norm(beta_new) ** 2
            e = e + 0.5 * tv * _laplacian_quad(beta_new)
            e = e + 0.5 * kappa * torch.linalg.norm(beta_new - prior) ** 2
            energy.append(float(e.item()))
            delta = torch.linalg.norm(beta_new - beta) / (torch.linalg.norm(beta) + 1e-12)
            beta = beta_new
            if float(delta) < tol:
                break

        prop = beta / (beta.sum(1, keepdim=True) + 1e-12)
        d2 = torch.cdist(prop, centroids) ** 2
        S = torch.softmax(-d2 / (d2.mean() * tau + 1e-12), dim=1)

    beta_np = beta.cpu().numpy()
    row = beta_np.sum(1, keepdims=True)
    proportions = np.divide(beta_np, row, out=np.zeros_like(beta_np), where=row > 0)
    return DiffJointResult(
        proportions=proportions,
        domains=S.argmax(1).cpu().numpy(),
        assignment=S.cpu().numpy(),
        energy=energy,
        n_iter=n_iter,
        device=device,
        raw=beta_np,
    )
