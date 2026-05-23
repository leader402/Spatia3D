"""Generic multi-slice spatial-transcriptomics stack loader (any AnnData-backed platform).

:func:`spatia3d.pipeline.run_unified` takes a ``coords_list`` + ``expression_list`` over a *shared*
gene set; real data instead arrives as a stack of per-slice AnnData (Visium, Stereo-seq, MERFISH,
Slide-seq all export this). This module is the glue between the two: read the slices, intersect them
to their common genes, optionally normalise (library-size + ``log1p``) and keep the top highly
variable genes, and emit a :class:`SpatialStack` whose ``coords_list`` / ``expression_list`` feed
the pipeline directly. So any dataset shipped as ``.h5ad`` becomes a Spatia3D benchmark in one call.

Dependency-light: numpy does the maths; ``anndata`` is imported lazily only to read ``.h5ad`` paths
(pass already-loaded AnnData objects and it is not needed at all).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = ["SpatialStack", "load_spatial_stack"]


@dataclass
class SpatialStack:
    """A gene-harmonised stack of spatial slices, ready for the unified pipeline."""

    coords_list: list[np.ndarray]  # per-slice (n_spots, 2) in-plane coordinates
    expression_list: list[np.ndarray]  # per-slice (n_spots, n_genes), shared gene order
    gene_names: list[str]
    slice_ids: list[str]
    labels_list: list[np.ndarray] | None = None  # per-slice ground-truth domain labels, if present
    meta: dict = field(default_factory=dict)

    @property
    def n_slices(self) -> int:
        return len(self.coords_list)

    @property
    def n_genes(self) -> int:
        return len(self.gene_names)

    def pipeline_inputs(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """The ``(coords_list, expression_list)`` pair to splat into :func:`run_unified`."""
        return self.coords_list, self.expression_list


def _to_anndata(src):
    """Accept an AnnData or a path to a ``.h5ad`` and return an AnnData."""
    if isinstance(src, (str, Path)):
        import anndata as ad

        return ad.read_h5ad(Path(src))
    return src  # assume already an AnnData-like object with .X, .var_names, .obsm, .obs


def _dense(X) -> np.ndarray:
    """Densify a (possibly sparse) AnnData ``.X`` to a float32 ndarray."""
    arr = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    return np.asarray(arr, dtype=np.float32)


def _normalize(X: np.ndarray, target_sum: float | None, log1p: bool) -> np.ndarray:
    """Per-spot library-size normalisation (to ``target_sum`` or median depth) + optional log."""
    lib = X.sum(axis=1, keepdims=True)
    target = target_sum if target_sum is not None else float(np.median(lib[lib > 0])) or 1.0
    Xn = np.divide(X, lib, out=np.zeros_like(X), where=lib > 0) * target
    return np.log1p(Xn) if log1p else Xn


def load_spatial_stack(
    sources,
    *,
    spatial_key: str = "spatial",
    label_col: str | None = None,
    normalize: bool = True,
    target_sum: float | None = None,
    log1p: bool = True,
    n_top_genes: int | None = None,
    slice_ids: list[str] | None = None,
) -> SpatialStack:
    """Assemble a stack of spatial slices into shared-gene pipeline inputs.

    Parameters
    ----------
    sources
        Iterable of slices, each an ``AnnData`` or a path to a ``.h5ad`` file.
    spatial_key
        ``obsm`` key holding spatial coordinates (first two columns are used as in-plane x/y).
    label_col
        Optional ``obs`` column with ground-truth domain labels; collected into ``labels_list``.
    normalize, target_sum, log1p
        Per-spot library-size normalisation (to ``target_sum`` or the per-slice median depth),
        followed by ``log1p``. Set ``normalize=False`` to keep raw values.
    n_top_genes
        If given, restrict to the top-``n_top_genes`` most variable genes (variance over the pooled,
        normalised stack) — the common HVG step before deconvolution / domain learning.
    slice_ids
        Optional names per slice (defaults to ``"slice{i}"`` or the AnnData ``uns["sample"]``).

    Returns
    -------
    SpatialStack
        With ``coords_list`` / ``expression_list`` over the shared (optionally HVG) gene set.
    """
    adatas = [_to_anndata(s) for s in sources]
    if not adatas:
        raise ValueError("sources is empty")

    # Common gene set, preserving the first slice's order.
    common = set(map(str, adatas[0].var_names))
    for a in adatas[1:]:
        common &= set(map(str, a.var_names))
    if not common:
        raise ValueError("no genes shared across all slices")
    genes = [g for g in map(str, adatas[0].var_names) if g in common]

    expr_list, coords_list, labels_list, ids = [], [], [], []
    for i, a in enumerate(adatas):
        names = list(map(str, a.var_names))
        idx = [names.index(g) for g in genes]  # align every slice to the shared gene order
        X = _dense(a.X)[:, idx]
        if normalize:
            X = _normalize(X, target_sum, log1p)
        expr_list.append(X)

        if spatial_key not in a.obsm:
            raise KeyError(f"slice {i}: obsm[{spatial_key!r}] missing (keys: {list(a.obsm)})")
        coords_list.append(np.asarray(a.obsm[spatial_key], dtype=float)[:, :2])

        if label_col is not None:
            if label_col not in a.obs:
                raise KeyError(f"slice {i}: obs[{label_col!r}] missing")
            labels_list.append(np.asarray(a.obs[label_col]))
        ids.append(
            slice_ids[i] if slice_ids is not None else str(a.uns.get("sample", f"slice{i}"))
        )

    if n_top_genes is not None and n_top_genes < len(genes):
        var = np.var(np.vstack(expr_list), axis=0)  # dispersion over the pooled stack
        keep = np.sort(np.argsort(var)[::-1][:n_top_genes])
        expr_list = [X[:, keep] for X in expr_list]
        genes = [genes[j] for j in keep]

    return SpatialStack(
        coords_list=coords_list,
        expression_list=expr_list,
        gene_names=genes,
        slice_ids=ids,
        labels_list=labels_list or None,
        meta={"n_slices": len(adatas), "normalized": normalize, "log1p": log1p and normalize},
    )
