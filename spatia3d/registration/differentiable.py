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


def _mean_sqdist(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Feature-mean pairwise squared distance ``||a-b||²/d`` via the Gram identity.

    Uses ``||a||² + ||b||² - 2 a·bᵀ`` so memory is ``O(m·n)``, never the ``O(m·n·d)`` tensor a naive
    broadcast would build — essential on real data (tens of thousands of genes would otherwise OOM).
    """
    a2 = (A * A).sum(1, keepdim=True)  # (m, 1)
    b2 = (B * B).sum(1).unsqueeze(0)  # (1, n)
    return (a2 + b2 - 2.0 * A @ B.t()).clamp_min(0.0) / A.shape[1]


def _control_grid(coords_centered, n_control, device):
    """Square grid of fixed RBF control points over the coord extent, plus spacing sigma."""
    allc = np.vstack(coords_centered)
    lo, hi = allc.min(0), allc.max(0)
    m = max(int(round(np.sqrt(n_control))), 2)
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], m), np.linspace(lo[1], hi[1], m))
    cp = np.column_stack([gx.ravel(), gy.ravel()])
    sigma = float(np.mean(hi - lo) / (m - 1))  # ~control-point spacing
    return torch.tensor(cp, dtype=torch.float64, device=device), sigma


def _rbf_field(x: torch.Tensor, cp: torch.Tensor, W: torch.Tensor, sigma: float) -> torch.Tensor:
    """Smooth differentiable displacement field: sum_k W_k * exp(-||x-cp_k||^2 / 2 sigma^2)."""
    d2 = torch.cdist(x, cp) ** 2
    return torch.exp(-d2 / (2 * sigma**2)) @ W


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
    anchor_weight: float = 0.0,
    nonrigid: bool = False,
    n_control: int = 16,
    device: str | None = None,
    seed: int = 0,
) -> OTAlignResult:
    """Refine slice alignment by gradient descent on a Sinkhorn-OT cost (spatial + expression).

    Each non-pivot slice gets a learnable rigid transform (angle + translation), init from ICP
    (``init="icp"``) or identity. With ``nonrigid=True`` a learnable smooth RBF deformation field
    (``n_control`` control points) is added on top, correcting non-rigid warps rigid cannot. The OT
    plan uses a combined cost (``alpha`` weights the spatial term); the loss is the plan-weighted
    spatial distance. ``anchor_weight`` (default 0) optionally ties each transform softly to its ICP
    init (a trust region): the OT objective is shallow and has spurious optima on near-aligned
    serial sections with changing morphology, where unconstrained OT can drift and worsen alignment
    vs ICP — a positive ``anchor_weight`` trades refinement power for that robustness (use it when
    the sections are already roughly aligned). Differentiable end-to-end (GPU when available).
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

    # Centre AND scale-normalise coords. The OT loss is plan-weighted ``C_sp`` (~coord²), so on real
    # pixel/array coordinates (DLPFC ~10²–10³) the gradient explodes and Adam diverges — alignment
    # then *worsens* the data; it only worked on the small-coordinate (~10) simulator. Optimise in
    # unit-scale space (lr/epsilon are tuned for it) and scale the outputs back.
    centered = [c - c.mean(0) for c in coords]
    scale = float(np.mean([np.sqrt((c**2).sum(1).mean()) for c in centered])) or 1.0
    Xc = [torch.tensor(c / scale, dtype=torch.float64, device=device) for c in centered]
    Ft = [
        torch.tensor((f - f.mean(0)) / (f.std(0) + 1e-8), dtype=torch.float64, device=device)
        for f in feats
    ]
    Xp = Xc[pivot]
    # Precompute fixed feature costs to the pivot per slice.
    Cf = [_mean_sqdist(Ft[s], Ft[pivot]) for s in range(n_slices)]

    cp, sigma = (
        _control_grid([c.cpu().numpy() for c in Xc], n_control, device) if nonrigid else (None, 1.0)
    )
    angles, transl, warp = {}, {}, {}
    params = []
    for s in range(n_slices):
        if s == pivot:
            continue
        a = torch.tensor(init_angles[s], dtype=torch.float64, device=device, requires_grad=True)
        t = torch.tensor(init_trans[s] / scale, dtype=torch.float64, device=device,
                         requires_grad=True)  # ICP translation is in original units
        angles[s], transl[s] = a, t
        params += [a, t]
        if nonrigid:
            warp[s] = torch.zeros(
                cp.shape[0], 2, dtype=torch.float64, device=device, requires_grad=True
            )
            params.append(warp[s])

    def transform(s):
        Xs = Xc[s] @ _rot(angles[s]).t() + transl[s]
        if nonrigid:
            Xs = Xs + _rbf_field(Xs, cp, warp[s], sigma)
        return Xs

    # Anchor each transform softly to its ICP init (trust region). The OT objective is only a proxy
    # for alignment and has spurious optima when serial sections differ in morphology (real embryo /
    # near-aligned slices), where unconstrained OT drifts and *worsens* the alignment vs ICP. The
    # anchor keeps OT near the good ICP init while still allowing it to refine genuine deformations.
    init_t = {s: torch.tensor(init_trans[s] / scale, dtype=torch.float64, device=device)
              for s in angles}

    opt = torch.optim.Adam(params, lr=lr)
    history: list[float] = []
    for _ in range(epochs):
        opt.zero_grad()
        total = 0.0
        for s in range(n_slices):
            if s == pivot:
                continue
            Xs = transform(s)
            C_sp = ((Xs[:, None, :] - Xp[None, :, :]) ** 2).sum(-1)
            cost = alpha * (C_sp / (C_sp.detach().mean() + 1e-12)) + (1 - alpha) * (
                Cf[s] / (Cf[s].mean() + 1e-12)
            )
            with torch.no_grad():
                plan = sinkhorn(cost, epsilon=epsilon, iters=sinkhorn_iters)
            total = total + (plan * C_sp).sum()
            if anchor_weight > 0:
                da = angles[s] - init_angles[s]
                dt = transl[s] - init_t[s]
                total = total + anchor_weight * (da * da + (dt * dt).sum())
        total.backward()
        opt.step()
        history.append(float(total.item()))

    aligned, out_angles, out_trans = [], np.zeros(n_slices), np.zeros((n_slices, 2))
    with torch.no_grad():
        for s in range(n_slices):
            if s == pivot:
                aligned.append(coords[s] - coords[s].mean(0))
                continue
            Xs = transform(s)
            aligned.append(Xs.cpu().numpy() * scale)  # back to original coordinate units
            out_angles[s] = float(angles[s])
            out_trans[s] = transl[s].cpu().numpy() * scale
    return OTAlignResult(
        aligned=aligned,
        angles=out_angles,
        translations=out_trans,
        loss_history=history,
        device=device,
        pivot=pivot,
    )
