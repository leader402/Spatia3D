"""C3 — Cell-type deconvolution.

Inverse problem  Y = beta @ V + e  with Elastic Net + column-wise spatial-TV regularization (the
theory-backed ADMM form from OptiGraph3D), made scalable: the small Gram matrix is diagonalised so
the P-update decouples into sparse SPD systems solvable by conjugate gradient instead of an explicit
matrix inverse (target: Stereo-seq million-spot).

Available: deconvolve_admm, DeconvolutionResult, spatial_laplacian, proportion_uncertainty
(per-spot Laplace posterior SD, matching the uncertainty RCTD/cell2location report). The unrolled
solver for C1 joint optimization lives in ``spatia3d.joint``.
Baselines wrapper: RCTD, cell2location, CARD, SONAR, stereoscope.
"""

from spatia3d.deconvolution.admm import DeconvolutionResult, deconvolve_admm
from spatia3d.deconvolution.graph import spatial_laplacian
from spatia3d.deconvolution.uncertainty import proportion_uncertainty

__all__ = [
    "deconvolve_admm",
    "DeconvolutionResult",
    "spatial_laplacian",
    "proportion_uncertainty",
]
