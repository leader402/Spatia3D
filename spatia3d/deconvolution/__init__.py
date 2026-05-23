"""C3 — Cell-type deconvolution.

Inverse problem  Y = beta @ V + e  with Elastic Net + column-wise spatial-TV regularization (the
theory-backed ADMM form from OptiGraph3D), made scalable: the small Gram matrix is diagonalised so
the P-update decouples into sparse SPD systems solvable by conjugate gradient instead of an explicit
matrix inverse (target: Stereo-seq million-spot).

Available: deconvolve_admm, DeconvolutionResult, spatial_laplacian.
TODO(M4): unrolled_solver() (for C1 joint optimization), uncertainty().
Baselines wrapper: RCTD, cell2location, CARD, SONAR, stereoscope.
"""

from spatia3d.deconvolution.admm import DeconvolutionResult, deconvolve_admm
from spatia3d.deconvolution.graph import spatial_laplacian

__all__ = ["deconvolve_admm", "DeconvolutionResult", "spatial_laplacian"]
