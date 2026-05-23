"""C1+C2 end-to-end joint model — registration co-trained with deconvolution + domains (torch/GPU).

The pieces so far run sequentially (align, then deconvolve<->domain). Here they share one gradient
flow: per-slice rigid transforms build a **differentiable soft 3D graph** (Gaussian affinities of
the transformed coordinates) feeding unrolled deconvolution + a soft-domain block, *and* a
differentiable OT alignment term. Both the OT term and the graph back-propagate to the transforms,
so registration is co-trained with the downstream tasks ("registration in the gradient flow").
NOTE: downstream smoothness *alone* is under-determined (admits spurious rigid configs that minimise
it without recovering geometry) — the OT term is what pins the alignment; the downstream refines it.

Distribution-space deconvolution (row-normalised Y, V) for scale stability; soft k-means domains.
Dense soft graph suits simulator scale; sparse/local affinities for large stacks (TODO).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import ArrayLike
from torch import nn

__all__ = ["JointEndToEnd", "fit_joint_end_to_end", "JointE2EResult"]


@dataclass
class JointE2EResult:
    proportions: np.ndarray
    domains: np.ndarray
    aligned_coords: list[np.ndarray]  # per-slice 2D in the joint frame
    loss_history: list[float]
    device: str = "cpu"
    angles: np.ndarray = field(default=None)


def _rot(theta):
    c, s = torch.cos(theta), torch.sin(theta)
    return torch.stack([torch.stack([c, -s]), torch.stack([s, c])])


class JointEndToEnd(nn.Module):
    """Learnable per-slice rigid transforms feeding a differentiable soft 3D graph + unrolled
    deconvolution + soft k-means domains."""

    def __init__(self, n_slices, n_celltypes, n_domains, pivot, *, n_steps=12, sigma=2.0):
        super().__init__()
        self.n_slices, self.pivot, self.n_steps = n_slices, pivot, n_steps
        self.n_domains = n_domains
        self.sigma = sigma
        self.angle = nn.ParameterList(
            [nn.Parameter(torch.zeros(())) for _ in range(n_slices)]
        )
        self.trans = nn.ParameterList(
            [nn.Parameter(torch.zeros(2)) for _ in range(n_slices)]
        )
        self.log_eta = nn.Parameter(torch.tensor(-1.0))
        self.log_tv = nn.Parameter(torch.tensor(-1.0))
        self.log_kappa = nn.Parameter(torch.tensor(-1.0))

    def init_transforms(self, angles, trans):
        with torch.no_grad():
            for s in range(self.n_slices):
                self.angle[s].copy_(torch.tensor(float(angles[s])))
                self.trans[s].copy_(torch.tensor(np.asarray(trans[s], dtype=float)))

    def aligned(self, coords_list):
        out = []
        for s, c in enumerate(coords_list):
            if s == self.pivot:
                out.append(c - c.mean(0))
            else:
                out.append((c - c.mean(0)) @ _rot(self.angle[s]).t() + self.trans[s])
        return out

    def soft_graph(self, R):
        # Gaussian affinity over 3D coords, row-normalised; differentiable in R (hence transforms).
        d2 = torch.cdist(R, R) ** 2
        A = torch.exp(-d2 / (2 * self.sigma**2))
        return A / A.sum(1, keepdim=True)

    def forward(self, coords_list, z, Yn, Vn):
        aligned = self.aligned(coords_list)
        R = torch.cat([torch.cat([a, z[s]], dim=1) for s, a in enumerate(aligned)], dim=0)
        A = self.soft_graph(R)
        eta = F.softplus(self.log_eta)
        tv, kappa = F.softplus(self.log_tv), F.softplus(self.log_kappa)

        beta = F.relu(Yn @ Vn.t())
        beta = beta / (beta.sum(1, keepdim=True) + 1e-8)
        idx = torch.linspace(0, beta.shape[0] - 1, self.n_domains).long()
        centroids = beta[idx].clone()  # spread-out init for soft k-means
        for _ in range(self.n_steps):
            d2 = torch.cdist(beta, centroids) ** 2
            S = torch.softmax(-d2 / (d2.mean() + 1e-9), dim=1)
            centroids = (S.t() @ beta) / (S.sum(0)[:, None] + 1e-8)
            prior = S @ centroids
            grad = (beta @ Vn - Yn) @ Vn.t() + tv * (beta - A @ beta) + kappa * (beta - prior)
            beta = F.relu(beta - eta * grad)
            beta = beta / (beta.sum(1, keepdim=True) + 1e-8)
        return beta, S, A, aligned


def fit_joint_end_to_end(
    coords_list: list[ArrayLike],
    expr_list: list[ArrayLike],
    V: ArrayLike,
    *,
    n_domains: int,
    pivot: int | None = None,
    z_spacing: float = 1.0,
    sigma: float = 2.0,
    n_steps: int = 12,
    epochs: int = 300,
    lr: float = 0.02,
    tv_weight: float = 1.0,
    kappa_weight: float = 1.0,
    align_weight: float = 5.0,
    alpha: float = 0.7,
    epsilon: float = 0.05,
    sinkhorn_iters: int = 50,
    init: str = "icp",
    device: str | None = None,
    seed: int = 0,
) -> JointE2EResult:
    """Co-train rigid transforms + unrolled deconvolution + soft domains end to end.

    Loss = data fidelity + tv·graph-smoothness + kappa·domain-coupling; the smoothness term flows
    gradients into the transforms (registration learned from downstream consistency). ``init="icp"``
    warm-starts the transforms from coarse ICP.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    n_slices = len(coords_list)
    pivot = n_slices // 2 if pivot is None else pivot
    coords = [np.asarray(c, dtype=np.float64) for c in coords_list]

    init_ang = np.zeros(n_slices)
    init_tr = np.zeros((n_slices, 2))
    if init == "icp":
        from spatia3d.registration import pivot_register

        _, tfs = pivot_register(coords, pivot=pivot, method="icp", max_iter=200)
        for s, (Rm, t) in enumerate(tfs):
            init_ang[s] = math.atan2(Rm[1, 0], Rm[0, 0])
            init_tr[s] = t

    Y = np.vstack([np.asarray(e, float) for e in expr_list])
    Vn = np.asarray(V, float)
    Yn = (Y + 1e-9) / (Y + 1e-9).sum(1, keepdims=True)
    Vn = (Vn + 1e-9) / (Vn + 1e-9).sum(1, keepdims=True)

    co_t = [torch.tensor(c, dtype=torch.float32, device=device) for c in coords]
    z_t = [torch.full((c.shape[0], 1), s * z_spacing, dtype=torch.float32, device=device)
           for s, c in enumerate(coords)]
    Yt = torch.tensor(Yn, dtype=torch.float32, device=device)
    Vt = torch.tensor(Vn, dtype=torch.float32, device=device)

    # Standardised features per slice + feature-cost to pivot, for the OT alignment term that pins
    # the geometry (downstream smoothness alone is under-determined and admits spurious transforms).
    from spatia3d.registration.differentiable import sinkhorn

    Ft = [
        torch.tensor((np.asarray(e, float) - np.asarray(e, float).mean(0))
                     / (np.asarray(e, float).std(0) + 1e-8), dtype=torch.float32, device=device)
        for e in expr_list
    ]
    Cf = {s: ((Ft[s][:, None, :] - Ft[pivot][None, :, :]) ** 2).mean(-1)
          for s in range(n_slices) if s != pivot}

    model = JointEndToEnd(
        n_slices, Vt.shape[0], n_domains, pivot, n_steps=n_steps, sigma=sigma
    ).to(device)
    model.init_transforms(init_ang, init_tr)
    # Optimise transforms (non-pivot) + unrolled step params.
    params = [model.log_eta, model.log_tv, model.log_kappa]
    for s in range(n_slices):
        if s != pivot:
            params += [model.angle[s], model.trans[s]]
    opt = torch.optim.Adam(params, lr=lr)

    history = []
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        beta, S, A, aligned = model(co_t, z_t, Yt, Vt)
        data = F.mse_loss(beta @ Vt, Yt)
        smooth = (beta * (beta - A @ beta)).sum() / beta.shape[0]
        couple = F.mse_loss(beta, S @ ((S.t() @ beta) / (S.sum(0)[:, None] + 1e-8)))
        # OT alignment term (pins geometry): plan-weighted spatial cost to the pivot per slice.
        align = beta.new_zeros(())
        Xp = aligned[pivot]
        for s in range(n_slices):
            if s == pivot:
                continue
            Csp = torch.cdist(aligned[s], Xp) ** 2
            cost = alpha * (Csp / (Csp.detach().mean() + 1e-9)) + (1 - alpha) * (
                Cf[s] / (Cf[s].mean() + 1e-9)
            )
            with torch.no_grad():
                plan = sinkhorn(cost.double(), epsilon=epsilon, iters=sinkhorn_iters).float()
            align = align + (plan * Csp).sum()
        loss = data + tv_weight * smooth + kappa_weight * couple + align_weight * align
        loss.backward()
        opt.step()
        history.append(float(loss.item()))

    model.eval()
    with torch.no_grad():
        beta, S, _, aligned = model(co_t, z_t, Yt, Vt)
    return JointE2EResult(
        proportions=beta.cpu().numpy(),
        domains=S.argmax(1).cpu().numpy(),
        aligned_coords=[a.cpu().numpy() for a in aligned],
        loss_history=history,
        device=device,
        angles=np.array([model.angle[s].detach().cpu().item() for s in range(n_slices)]),
    )
