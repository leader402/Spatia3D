"""Dataset loaders + simulators (see docs/PLAN.md 2.1).

Real: DLPFC (spatialLIBD, Visium) — all 12 samples via Figshare h5ad — plus a *platform-agnostic*
stack loader (``load_spatial_stack``) that turns any ``.h5ad`` slices (Stereo-seq, MERFISH,
Slide-seq, ...) into pipeline inputs over a shared gene set.
Simulation: custom generator with known proportions + domains + 3D deformation (ground truth).

Available: simulate_3d (ground-truth bench), load_dlpfc / download_dlpfc (151507 Space Ranger),
download_dlpfc_all / load_dlpfc_h5ad (full 12 samples, Figshare), load_spatial_stack / SpatialStack
(generic multi-slice loader).
"""

from spatia3d.datasets.dlpfc import (
    DLPFC_SAMPLES,
    download_dlpfc,
    download_dlpfc_all,
    load_dlpfc,
    load_dlpfc_h5ad,
)
from spatia3d.datasets.heart import (
    HEART_STAGES,
    download_heart,
    load_heart,
    load_heart_reference,
)
from spatia3d.datasets.simulate import (
    SimulatedDataset,
    SingleCellReference,
    Slice,
    simulate_3d,
)
from spatia3d.datasets.stack import SpatialStack, load_spatial_stack

__all__ = [
    "Slice",
    "SimulatedDataset",
    "SingleCellReference",
    "simulate_3d",
    "load_dlpfc",
    "download_dlpfc",
    "download_dlpfc_all",
    "load_dlpfc_h5ad",
    "DLPFC_SAMPLES",
    "SpatialStack",
    "load_spatial_stack",
    "load_heart",
    "load_heart_reference",
    "download_heart",
    "HEART_STAGES",
]
