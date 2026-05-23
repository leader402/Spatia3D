"""Synthetic 3D spatial-transcriptomics simulator with full ground truth.

Generates a stack of parallel slices through a layered 3D tissue where, for every spot, we know the
*true* cell-type proportions (beta), the *true* spatial domain, the canonical (undeformed) 3D
coordinate, and the per-slice deformation that was applied. Expression follows the OptiGraph3D
forward model ``Y = beta @ V + noise`` with a structured reference signature ``V``.

This is the ground-truth bench for the three coupled tasks (PLAN.md 2.1):
  * deconvolution  -> recover beta from (Y, V); score with RMSE / JSD / PCC.
  * domain id      -> recover domains from (Y, coords); score with ARI / NMI.
  * registration   -> recover canonical coords from deformed observations; score landmark error.

Dependency-light: needs only numpy. ``to_anndata`` is provided for when scanpy/anndata are present.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["Slice", "SimulatedDataset", "SingleCellReference", "simulate_3d"]


@dataclass
class Slice:
    """One physical section through the simulated volume."""

    z: float
    coords_true: np.ndarray  # (n_spots, 2) canonical in-plane coords (aligned across slices)
    coords_obs: np.ndarray  # (n_spots, 2) observed coords after the slice's deformation
    rotation: np.ndarray  # (2, 2) applied rotation matrix
    translation: np.ndarray  # (2,) applied translation
    expression: np.ndarray  # (n_spots, n_genes)
    proportions: np.ndarray  # (n_spots, n_celltypes) ground-truth beta (rows sum to 1)
    domains: np.ndarray  # (n_spots,) int domain labels

    @property
    def n_spots(self) -> int:
        return self.coords_true.shape[0]


@dataclass
class SingleCellReference:
    """Synthetic annotated scRNA reference for deconvolution gold standards to learn from."""

    counts: np.ndarray  # (n_cells, n_genes)
    labels: np.ndarray  # (n_cells,) integer cell-type index
    celltype_names: list[str]

    def signature_estimate(self) -> np.ndarray:
        """Mean expression profile per cell type, ``(n_celltypes, n_genes)`` — should track V."""
        K = len(self.celltype_names)
        return np.vstack([self.counts[self.labels == c].mean(axis=0) for c in range(K)])


@dataclass
class SimulatedDataset:
    """A stack of slices plus the shared reference signature and metadata."""

    slices: list[Slice]
    signatures: np.ndarray  # (n_celltypes, n_genes) reference V
    celltype_names: list[str]
    domain_compositions: np.ndarray  # (n_domains, n_celltypes) mean composition per domain
    meta: dict = field(default_factory=dict)

    # --- concatenated views across the whole stack -------------------------------------------
    def _stack(self, attr: str) -> np.ndarray:
        return np.concatenate([getattr(s, attr) for s in self.slices], axis=0)

    def expression(self) -> np.ndarray:
        return self._stack("expression")

    def proportions(self) -> np.ndarray:
        return self._stack("proportions")

    def domains(self) -> np.ndarray:
        return self._stack("domains")

    def coords_obs(self) -> np.ndarray:
        return self._stack("coords_obs")

    def coords_true_3d(self) -> np.ndarray:
        """Canonical (x, y, z) coordinates — the alignment target."""
        return np.concatenate(
            [np.column_stack([s.coords_true, np.full(s.n_spots, s.z)]) for s in self.slices],
            axis=0,
        )

    def slice_ids(self) -> np.ndarray:
        return np.concatenate([np.full(s.n_spots, i, dtype=int) for i, s in enumerate(self.slices)])

    def single_cell_reference(
        self, *, n_cells_per_type: int = 200, count_scale: float = 300.0, seed: int = 0
    ) -> SingleCellReference:
        """Emit a synthetic annotated scRNA reference consistent with the signatures ``V``.

        Each cell's counts are Poisson-sampled around its cell type's signature row (scaled to
        single-cell depth). Gold standards (cell2location, RCTD, stereoscope, STitch3D) learn
        signatures from such a reference; because it shares ``V`` with the spatial data, it is
        the matched-reference setting standard deconvolution benchmarks use.
        """
        rng = np.random.default_rng(seed)
        K = self.signatures.shape[0]
        counts, labels = [], []
        for c in range(K):
            mean = np.maximum(self.signatures[c] * count_scale, 0.0)
            counts.append(rng.poisson(mean, size=(n_cells_per_type, mean.shape[0])).astype(float))
            labels.extend([c] * n_cells_per_type)
        counts = np.vstack(counts)
        labels = np.array(labels)
        perm = rng.permutation(counts.shape[0])
        return SingleCellReference(counts[perm], labels[perm], list(self.celltype_names))

    def to_anndata(self):  # pragma: no cover - exercised only when anndata is installed
        """Export to an AnnData with obs metadata and ground truth in obsm/uns."""
        try:
            import anndata as ad
            import pandas as pd
        except ImportError as exc:  # pragma: no cover
            raise ImportError("to_anndata requires `anndata` and `pandas`") from exc
        obs = pd.DataFrame(
            {
                "slice": self.slice_ids().astype(str),
                "domain": self.domains().astype(str),
            }
        )
        adata = ad.AnnData(X=self.expression().astype(np.float32), obs=obs)
        adata.obsm["spatial"] = self.coords_obs()
        adata.obsm["spatial_true_3d"] = self.coords_true_3d()
        adata.obsm["proportions_true"] = self.proportions()
        adata.uns["signatures"] = self.signatures
        adata.uns["celltype_names"] = self.celltype_names
        return adata


def _make_signatures(
    rng: np.random.Generator, n_celltypes: int, n_genes: int, markers_per_celltype: int
) -> np.ndarray:
    """Reference matrix V (n_celltypes, n_genes): low baseline + per-type marker boosts."""
    V = rng.uniform(0.05, 0.2, size=(n_celltypes, n_genes))
    n_markers = min(markers_per_celltype, n_genes)
    for c in range(n_celltypes):
        markers = rng.choice(n_genes, size=n_markers, replace=False)
        V[c, markers] += rng.uniform(1.0, 3.0, size=n_markers)
    return V


def _make_domain_compositions(
    rng: np.random.Generator, n_domains: int, n_celltypes: int
) -> np.ndarray:
    """Each domain favours one or two dominant cell types over a low background."""
    comps = np.empty((n_domains, n_celltypes))
    for d in range(n_domains):
        alpha = np.full(n_celltypes, 0.3)
        dominant = rng.choice(n_celltypes, size=min(2, n_celltypes), replace=False)
        alpha[dominant] += rng.uniform(3.0, 6.0, size=dominant.size)
        comps[d] = rng.dirichlet(alpha)
    return comps


def _assign_domains(
    coords: np.ndarray, z_frac: float, n_domains: int, side: float, tilt: float
) -> np.ndarray:
    """Layered domains along y; ``tilt`` shifts the bands with depth for genuine 3D structure."""
    y_frac = coords[:, 1] / side
    shifted = np.clip(y_frac + tilt * z_frac, 0.0, 1.0 - 1e-9)
    return np.floor(shifted * n_domains).astype(int)


def simulate_3d(
    *,
    n_slices: int = 6,
    spots_per_side: int = 20,
    n_celltypes: int = 8,
    n_genes: int = 200,
    n_domains: int = 5,
    markers_per_celltype: int = 15,
    composition_concentration: float = 50.0,
    domain_tilt: float = 0.0,
    noise: float = 0.1,
    count_model: str = "gaussian",
    library_size: float = 200.0,
    max_rotation_deg: float = 15.0,
    max_translation: float = 2.0,
    pivot_identity: bool = True,
    seed: int = 0,
) -> SimulatedDataset:
    """Simulate a 3D ST stack with known proportions, domains, and per-slice deformations.

    Parameters
    ----------
    n_slices, spots_per_side
        Stack depth and the (square) per-slice spot grid (``spots_per_side**2`` spots/slice).
    n_celltypes, n_genes, n_domains, markers_per_celltype
        Sizes of the reference signature and the domain layout.
    composition_concentration
        Dirichlet concentration for per-spot beta around its domain mean — larger is tighter
        (more recoverable, more domain-coherent).
    domain_tilt
        Shift of domain bands per unit depth; ``0`` is a clean layered slab (DLPFC-like), ``>0``
        tilts layers across z for a 3D-native structure.
    noise, count_model, library_size
        Expression noise level; ``"gaussian"`` (additive, continuous) or ``"poisson"`` (counts);
        overall expression scale.
    max_rotation_deg, max_translation, pivot_identity
        Random per-slice rigid deformation magnitude. With ``pivot_identity`` the middle slice is
        left undeformed, so aligning others to it equals aligning to the canonical frame.
    seed
        RNG seed for full reproducibility.
    """
    if count_model not in {"gaussian", "poisson"}:
        raise ValueError(f"count_model must be 'gaussian' or 'poisson', got {count_model!r}")
    rng = np.random.default_rng(seed)
    side = float(spots_per_side - 1)

    V = _make_signatures(rng, n_celltypes, n_genes, markers_per_celltype)
    domain_comps = _make_domain_compositions(rng, n_domains, n_celltypes)

    # Canonical grid shared by every slice (the alignment target).
    gx, gy = np.meshgrid(np.arange(spots_per_side), np.arange(spots_per_side))
    grid = np.column_stack([gx.ravel(), gy.ravel()]).astype(float)
    n_spots = grid.shape[0]

    pivot = n_slices // 2
    max_rot = np.deg2rad(max_rotation_deg)
    slices: list[Slice] = []
    for i in range(n_slices):
        z = float(i)
        z_frac = i / max(n_slices - 1, 1)

        domains = _assign_domains(grid, z_frac, n_domains, side, domain_tilt)

        # Per-spot beta: Dirichlet around the spot's domain mean composition.
        alpha = domain_comps[domains] * composition_concentration + 1e-6
        proportions = np.array([rng.dirichlet(a) for a in alpha])

        mean_expr = (proportions @ V) * library_size
        if count_model == "gaussian":
            sigma = noise * mean_expr.mean()
            expression = np.clip(mean_expr + rng.normal(0.0, sigma, size=mean_expr.shape), 0, None)
        else:  # poisson
            expression = rng.poisson(np.clip(mean_expr, 0, None)).astype(float)

        # Per-slice rigid deformation; the pivot slice may be left identity.
        if pivot_identity and i == pivot:
            theta, t = 0.0, np.zeros(2)
        else:
            theta = rng.uniform(-max_rot, max_rot)
            t = rng.uniform(-max_translation, max_translation, size=2)
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        coords_obs = grid @ R.T + t

        slices.append(
            Slice(
                z=z,
                coords_true=grid.copy(),
                coords_obs=coords_obs,
                rotation=R,
                translation=t,
                expression=expression,
                proportions=proportions,
                domains=domains,
            )
        )

    meta = {
        "n_slices": n_slices,
        "n_spots_per_slice": n_spots,
        "n_celltypes": n_celltypes,
        "n_genes": n_genes,
        "n_domains": n_domains,
        "pivot": pivot,
        "count_model": count_model,
        "seed": seed,
    }
    return SimulatedDataset(
        slices=slices,
        signatures=V,
        celltype_names=[f"celltype_{c}" for c in range(n_celltypes)],
        domain_compositions=domain_comps,
        meta=meta,
    )
