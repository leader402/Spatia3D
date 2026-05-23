"""C2 — differentiable optimal-transport slice alignment (torch, GPU).

Coarse ICP/Kabsch (``rigid.py``/``pivot.py``) gives a fast init; this module then *refines* each
slice's transform by gradient descent on an entropic-OT (Sinkhorn) alignment cost combining spatial
distance (after the learnable transform) with expression dissimilarity. Because every step is
differentiable, alignment error flows into the gradient and can be optimised jointly with the
downstream deconvolution / domain model — the property the unrolled C1 was built to exploit, and the
basis for the non-rigid / cross-platform alignment the field now expects.

A learnable rigid transform per (non-pivot) slice is provided here; a control-point displacement
field (non-rigid) plugs into the same loss (TODO).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
from numpy.typing import ArrayLike

__all__ = ["sinkhorn", "differentiable_ot_align", "OTAlignResult"]


@dataclass
class OTAlignResult:
    aligned: list[np.ndarray]  # coordinates per slice in the pivot frame
    angles: np.ndarray  # learned rotation angle per slice (radians; 0 for pivot)
    translations: np.ndarray  # learned translation per slice
    loss_history: list[float]
    device: str = "cpu"
    pivot: int = field(default=0)


def sinkhorn(C: torch.Tensor, epsilon: float = 0.1, iters: int = 50) -> torch.Tensor:
    """Entropic-OT transport plan for cost ``C``, uniform marginals (log-domain, differentiable)."""
    m, n = C.shape
    log_mu = torch.full((m,), -math.log(m), device=C.device, dtype=C.dtype)
    log_nu = torch.full((n,), -math.log(n), device=C.device, dtype=C.dtype)
    Ke = -C / epsilon
    u = torch.zeros(m, device=C.device, dtype=C.dtype)
    v = torch.zeros(n, device=C.device, dtype=C.dtype)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Ke + v[None, :], dim=1)
        v = log_nu - torch.logsumexp(Ke + u[:, None], dim=0)
    return torch.exp(Ke + u[:, None] + v[None, :])


def _rot(theta: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(theta), torch.sin(theta)
    return torch.stack([torch.stack([c, -s]), torch.stack([s, c])])


def differentiable_ot_align(
    coords_list: list[ArrayLike],
    features_list: list[ArrayLike],
    *,
    pivot: int | None = None,
    init: str = "icp",
    alpha: float = 0.7,
    epsilon: float = 0.05,
    sinkhorn_iters: int = 50,
    epochs: int = 150,
    lr: float = 0.05,
    device: str | None = None,
    seed: int = 0,
) -> OTAlignResult:
    """Refine slice alignment by gradient descent on a Sinkhorn-OT cost (spatial + expression).

    Each non-pivot slice gets a learnable rigid transform (angle + translation), init from ICP
    (``init="icp"``) or identity. The OT plan is computed on a combined cost (``alpha`` weights the
    spatial term); the loss is the plan-weighted spatial distance, so minimising it aligns
    feature-matched spots. Differentiable end-to-end (GPU when available).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    coords = [np.asarray(c, dtype=np.float64) for c in coords_list]
    feats = [np.asarray(f, dtype=np.float64) for f in features_list]
    n_slices = len(coords)
    pivot = n_slices // 2 if pivot is None else pivot

    # Coarse initialisation (handles gross rotation so OT refines from a good basin).
    init_angles = np.zeros(n_slices)
    init_trans = np.zeros((n_slices, 2))
    if init == "icp":
        from spatia3d.registration import pivot_register

        _, transforms = pivot_register(coords, pivot=pivot, method="icp", max_iter=200)
        for s, (R, t) in enumerate(transforms):
            init_angles[s] = math.atan2(R[1, 0], R[0, 0])
            init_trans[s] = t

    # Centre coords per slice (stability); features standardised for the cost.
    Xc = [torch.tensor(c - c.mean(0), dtype=torch.float64, device=device) for c in coords]
    Ft = [
        torch.tensor((f - f.mean(0)) / (f.std(0) + 1e-8), dtype=torch.float64, device=device)
        for f in feats
    ]
    Xp = Xc[pivot]
    # Precompute fixed feature costs to the pivot per slice.
    Cf = [((Ft[s][:, None, :] - Ft[pivot][None, :, :]) ** 2).mean(-1) for s in range(n_slices)]

    angles, transl = {}, {}
    params = []
    for s in range(n_slices):
        if s == pivot:
            continue
        a = torch.tensor(init_angles[s], dtype=torch.float64, device=device, requires_grad=True)
        t = torch.tensor(init_trans[s], dtype=torch.float64, device=device, requires_grad=True)
        angles[s], transl[s] = a, t
        params += [a, t]

    opt = torch.optim.Adam(params, lr=lr)
    history: list[float] = []
    for _ in range(epochs):
        opt.zero_grad()
        total = 0.0
        for s in range(n_slices):
            if s == pivot:
                continue
            Xs = Xc[s] @ _rot(angles[s]).t() + transl[s]
            C_sp = ((Xs[:, None, :] - Xp[None, :, :]) ** 2).sum(-1)
            cost = alpha * (C_sp / (C_sp.detach().mean() + 1e-12)) + (1 - alpha) * (
                Cf[s] / (Cf[s].mean() + 1e-12)
            )
            with torch.no_grad():
                plan = sinkhorn(cost, epsilon=epsilon, iters=sinkhorn_iters)
            total = total + (plan * C_sp).sum()
        total.backward()
        opt.step()
        history.append(float(total.item()))

    aligned, out_angles, out_trans = [], np.zeros(n_slices), np.zeros((n_slices, 2))
    with torch.no_grad():
        for s in range(n_slices):
            if s == pivot:
                aligned.append(coords[s] - coords[s].mean(0))
                continue
            Xs = Xc[s] @ _rot(angles[s]).t() + transl[s]
            aligned.append(Xs.cpu().numpy())
            out_angles[s] = float(angles[s])
            out_trans[s] = transl[s].cpu().numpy()
    return OTAlignResult(
        aligned=aligned,
        angles=out_angles,
        translations=out_trans,
        loss_history=history,
        device=device,
        pivot=pivot,
    )
