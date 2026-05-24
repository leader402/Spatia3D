"""Spatia3D unified pipeline — the full framework end to end.

Composes the contributions into one call: C2 differentiable OT alignment -> a true cross-slice 3D
graph -> C1 joint deconvolution<->domain optimization on that graph. The ``align`` and ``joint``
switches let the same function produce the ablation variants (no-alignment, staged pipeline, full
unified), so the headline claim — *the unified framework beats the staged pipeline* — is one
experiment over this single entry point.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike
from sklearn.cluster import KMeans

from spatia3d.deconvolution import deconvolve_admm
from spatia3d.joint import joint_deconvolution_domain

__all__ = ["UnifiedResult", "run_unified"]


@dataclass
class UnifiedResult:
    proportions: np.ndarray  # (n_spots_total, n_celltypes)
    domains: np.ndarray  # (n_spots_total,) 3D spatial-domain labels
    coords_3d: np.ndarray  # (n_spots_total, 3) aligned 3D coordinates
    slice_ids: np.ndarray  # (n_spots_total,)
    align_mode: str = "ot"
    joint_mode: str = "joint"
    energy: list[float] = field(default_factory=list)


def _aligned_coords(coords_list, expression_list, align, *, seed, device):
    if align == "none":
        return [np.asarray(c, float) - np.asarray(c, float).mean(0) for c in coords_list]
    if align == "icp":
        from spatia3d.registration import pivot_register

        aligned, _ = pivot_register([np.asarray(c, float) for c in coords_list], method="icp")
        return [a - a.mean(0) for a in aligned]
    if align == "ot":
        from spatia3d.registration import differentiable_ot_align

        return differentiable_ot_align(
            coords_list, expression_list, init="icp", epochs=120, device=device, seed=seed
        ).aligned
    raise ValueError(f"unknown align mode {align!r}")


def run_unified(
    coords_list: list[ArrayLike],
    expression_list: list[ArrayLike],
    V: ArrayLike,
    *,
    n_domains: int,
    align: str = "ot",
    joint: str = "joint",
    z_spacing: float = 1.0,
    k: int = 8,
    prior_weight: float = 15.0,  # the C1 domain-prior coupling strength; ~2 effectively disables it
    tv: float = 0.5,
    seed: int = 0,
    device: str | None = None,
) -> UnifiedResult:
    """Run the Spatia3D pipeline on a slice stack.

    ``joint`` selects how the tasks are coupled:

    * ``"separate"`` — align (per ``align``), then deconvolve, then cluster: a one-directional
      pipeline, no feedback (the baseline to beat).
    * ``"joint"`` — align (per ``align``), then C1 block-coordinate deconvolution<->domain on the
      cross-slice 3D graph (the two downstream tasks are mutual priors; alignment is staged).
    * ``"end_to_end"`` — all three tasks share one gradient flow: registration is co-trained with
      unrolled deconvolution + soft domains (C1+C2 together). ``align`` is ignored here, since the
      transforms are learned inside the model.

    ``align`` ∈ {``"ot"`` (differentiable OT, C2), ``"icp"``, ``"none"``} sets the alignment for the
    staged modes. Returns proportions + 3D domains over the (aligned) cross-slice coordinates; the
    ``joint``/``align`` switches make the unified-vs-staged ablation a single entry point.
    """
    Y = np.vstack([np.asarray(e, float) for e in expression_list])
    V = np.asarray(V, float)

    energy: list[float] = []
    if joint == "end_to_end":
        from spatia3d.joint import fit_joint_end_to_end

        res = fit_joint_end_to_end(
            coords_list, expression_list, V, n_domains=n_domains, z_spacing=z_spacing,
            seed=seed, device=device,
        )
        aligned = res.aligned_coords
        proportions, domains, energy = res.proportions, res.domains, res.loss_history
        align = "co-trained"
    else:
        aligned = _aligned_coords(coords_list, expression_list, align, seed=seed, device=device)
        coords_3d = _stack_3d(aligned, z_spacing)
        if joint == "joint":
            r = joint_deconvolution_domain(
                Y, V, coords=coords_3d, n_domains=n_domains, prior_weight=prior_weight, tv=tv,
                k=k, n_outer=12, seed=seed,
            )
            proportions, domains, energy = r.proportions, r.domains, r.energy
        elif joint == "separate":
            beta = deconvolve_admm(Y, V, coords=coords_3d, tv=tv, k=k, normalize=False).raw
            row = beta.sum(axis=1, keepdims=True)
            proportions = np.divide(beta, row, out=np.zeros_like(beta), where=row > 0)
            domains = KMeans(n_clusters=n_domains, n_init=10, random_state=seed).fit_predict(beta)
        else:
            raise ValueError(f"unknown joint mode {joint!r}")

    coords_3d = _stack_3d(aligned, z_spacing)
    slice_ids = np.concatenate([np.full(len(a), s, dtype=int) for s, a in enumerate(aligned)])
    return UnifiedResult(
        proportions=proportions,
        domains=domains,
        coords_3d=coords_3d,
        slice_ids=slice_ids,
        align_mode=align,
        joint_mode=joint,
        energy=energy,
    )


def _stack_3d(aligned, z_spacing):
    """Stack per-slice 2D coords into (n_total, 3), appending slice depth ``s · z_spacing``."""
    return np.vstack(
        [
            np.column_stack([np.asarray(a, float), np.full(len(a), s * z_spacing)])
            for s, a in enumerate(aligned)
        ]
    )
