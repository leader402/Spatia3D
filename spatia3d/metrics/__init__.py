"""Evaluation metrics (see docs/PLAN.md 2.3).

Domain: ARI, NMI, ASW.  Deconvolution: RMSE, JSD, PCC, AUPRC, rare-cell score.
Alignment: landmark error, spatial coherence, cross-slice label continuity.
3D: inter-layer continuity (= cross-slice continuity).  System metrics (runtime/memory) live with
the experiment harness, not here.
"""

from spatia3d.metrics.alignment import (
    cross_slice_label_continuity,
    label_transfer_accuracy,
    landmark_error,
    spatial_coherence_score,
)
from spatia3d.metrics.deconvolution import (
    auprc,
    credible_coverage,
    deconvolution_report,
    jsd,
    pcc,
    rare_cell_score,
    rmse,
)
from spatia3d.metrics.domain import ari, asw, nmi

__all__ = [
    # domain
    "ari",
    "nmi",
    "asw",
    # deconvolution
    "rmse",
    "jsd",
    "pcc",
    "auprc",
    "rare_cell_score",
    "credible_coverage",
    "deconvolution_report",
    # alignment / 3D
    "landmark_error",
    "label_transfer_accuracy",
    "spatial_coherence_score",
    "cross_slice_label_continuity",
]
