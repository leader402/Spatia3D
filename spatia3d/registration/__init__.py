"""C2 — Slice registration.

Pipeline: pivot-slice rigid coarse alignment (Kabsch / ICP; keeps OptiGraph3D's speed and
no-cumulative-error advantage) -> differentiable optimal-transport (Sinkhorn) refinement that feeds
gradients into downstream, so alignment can be optimised jointly with deconvolution / domains.
Handles partial overlap and cross-platform (Visium/Slide-seq/Stereo-seq/MERFISH/Xenium).

Available: kabsch, icp, apply_transform, pivot_register; differentiable_ot_align + sinkhorn (torch,
lazy-imported). TODO(M2): non-rigid control-point field; baselines wrapper for PASTE2/GPSA/CAST.
"""
from spatia3d.registration.pivot import pivot_register
from spatia3d.registration.rigid import apply_transform, icp, kabsch

__all__ = [
    "kabsch",
    "icp",
    "apply_transform",
    "pivot_register",
    "differentiable_ot_align",
    "sinkhorn",
    "OTAlignResult",
]


def __getattr__(name: str):
    # Lazy (torch-dependent) differentiable OT alignment; keeps the core package torch-free.
    if name in {"differentiable_ot_align", "sinkhorn", "OTAlignResult"}:
        from spatia3d.registration import differentiable

        return getattr(differentiable, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
