"""Dataset loaders + simulators (see docs/PLAN.md 2.1).

Real: DLPFC (spatialLIBD, Visium) — all 12 samples via Figshare h5ad; human embryonic heart,
mouse embryo MOSTA (Stereo-seq), Drosophila embryo, mouse brain (MERFISH/Xenium) — TODO.
Simulation: custom generator with known proportions + domains + 3D deformation (ground truth).

Available: simulate_3d (ground-truth bench), load_dlpfc / download_dlpfc (151507 Space Ranger),
download_dlpfc_all / load_dlpfc_h5ad (full 12 samples, Figshare). TODO: load_heart(), load_mosta().
"""

from spatia3d.datasets.dlpfc import (
    DLPFC_SAMPLES,
    download_dlpfc,
    download_dlpfc_all,
    load_dlpfc,
    load_dlpfc_h5ad,
)
from spatia3d.datasets.simulate import (
    SimulatedDataset,
    SingleCellReference,
    Slice,
    simulate_3d,
)

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
]
