"""C1 — unified joint optimization coupling deconvolution (C3) and spatial-domain id (C4).

The headline contribution: instead of a one-directional pipeline, deconvolution and domain
identification are solved jointly as block-coordinate descent on one coupled energy, with a monotone
(hence convergent) guarantee — formalising OptiGraph3D's mutual-supervision idea.
"""
from spatia3d.joint.optimizer import JointResult, joint_deconvolution_domain

__all__ = ["JointResult", "joint_deconvolution_domain"]


def __getattr__(name: str):
    # Lazy (torch-dependent) differentiable C1 variants; optimizer.py is the numpy primary.
    if name in {"DiffJointResult", "differentiable_joint_c1"}:
        from spatia3d.joint import unrolled

        return getattr(unrolled, name)
    if name in {"JointEndToEnd", "fit_joint_end_to_end", "JointE2EResult"}:
        from spatia3d.joint import end_to_end

        return getattr(end_to_end, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
