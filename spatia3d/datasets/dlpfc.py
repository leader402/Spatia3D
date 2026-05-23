"""DLPFC loader (spatialLIBD, 10x Visium) — the domain-identification main benchmark.

Loads a Space Ranger sample directory (``filtered_feature_bc_matrix.h5`` + ``metadata.tsv`` with
manual layer annotations + ``spatial/``) into an AnnData carrying expression, spatial coordinates,
and the ground-truth cortical-layer label. ``download_dlpfc`` fetches one sample (151507) from the
GraphST mirror — the only fully scriptable source; the full 12 samples come from the spatialLIBD
R/Bioconductor package or LieberInstitute/HumanPilot (see docs/PLAN.md 2.1).
"""
from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

__all__ = ["load_dlpfc", "download_dlpfc", "download_dlpfc_all", "load_dlpfc_h5ad", "DLPFC_SAMPLES"]

#: The 12 DLPFC samples (spatialLIBD).
DLPFC_SAMPLES = (
    "151507", "151508", "151509", "151510", "151669", "151670",
    "151671", "151672", "151673", "151674", "151675", "151676",
)

# Preprocessed per-sample .h5ad on Figshare (Visium DLPFC preprocessed, doi 22004273) — the only
# fully scriptable source for all 12 samples. obs has `sce.layer_guess`; obsm has `spatial`.
_FIGSHARE = {
    "151507": "39055556", "151508": "39055589", "151509": "39055586", "151510": "39055583",
    "151669": "39055580", "151670": "39055577", "151671": "39055574", "151672": "39055571",
    "151673": "39055568", "151674": "39055565", "151675": "39055562", "151676": "39055559",
}

# GraphST hosts this sample under a directory labelled "151673", but the file contents
# (sample_name=151507, subject Br5292/pos0/rep1, 4226 spots) are unambiguously sample 151507.
_GRAPHST_BASE = "https://raw.githubusercontent.com/JinmiaoChenLab/GraphST/main/Data/151673"
_GRAPHST_FILES = (
    "filtered_feature_bc_matrix.h5",
    "metadata.tsv",
    "spatial/tissue_positions_list.csv",
)


def download_dlpfc(dest_root: str | Path, sample: str = "151507") -> Path:
    """Download a DLPFC sample into ``dest_root/sample`` and return that path.

    Only sample ``151507`` is available from the scriptable GraphST mirror. Existing files are not
    re-downloaded.
    """
    if sample != "151507":
        raise ValueError(
            f"only '151507' is scriptable from the GraphST mirror; got {sample!r}. "
            "For the full 12 samples use the spatialLIBD R package or LieberInstitute/HumanPilot."
        )
    out = Path(dest_root) / sample
    (out / "spatial").mkdir(parents=True, exist_ok=True)
    for rel in _GRAPHST_FILES:
        target = out / rel
        if not target.exists():
            urlretrieve(f"{_GRAPHST_BASE}/{rel}", target)
    return out


def download_dlpfc_all(dest_root: str | Path, samples=DLPFC_SAMPLES) -> list[Path]:
    """Download the preprocessed ``.h5ad`` for the given DLPFC samples from Figshare (~110 MB each).

    Returns the per-sample paths; existing files are not re-downloaded. Load with
    :func:`load_dlpfc_h5ad`.
    """
    dest = Path(dest_root)
    dest.mkdir(parents=True, exist_ok=True)
    paths = []
    for s in samples:
        if s not in _FIGSHARE:
            raise ValueError(f"unknown DLPFC sample {s!r}")
        target = dest / f"{s}.h5ad"
        if not target.exists():
            urlretrieve(f"https://ndownloader.figshare.com/files/{_FIGSHARE[s]}", target)
        paths.append(target)
    return paths


def load_dlpfc_h5ad(
    path: str | Path, *, label_col: str = "sce.layer_guess", filter_to_labeled: bool = False
):
    """Load a preprocessed DLPFC ``.h5ad`` (Figshare format) into an AnnData.

    Sets ``obs["layer"]`` (ground truth) from ``label_col`` and ``uns["sample"]``; keeps the spatial
    coordinates in ``obsm["spatial"]`` and array indices in ``obs["array_row"/"array_col"]``.
    """
    import anndata as ad

    adata = ad.read_h5ad(Path(path))
    adata.var_names_make_unique()
    if label_col in adata.obs:
        adata.obs["layer"] = adata.obs[label_col].astype("category")
    elif "layer" not in adata.obs:
        raise KeyError(f"label column {label_col!r} not in obs: {list(adata.obs.columns)}")
    if "sce.sample_name" in adata.obs:
        adata.uns["sample"] = str(adata.obs["sce.sample_name"].iloc[0])
    if filter_to_labeled:
        adata = adata[adata.obs["layer"].notna()].copy()
    return adata


def load_dlpfc(
    sample_dir: str | Path,
    *,
    label_col: str = "layer_guess",
    coord_cols: tuple[str, str] = ("imagerow", "imagecol"),
    filter_to_labeled: bool = False,
):
    """Load a DLPFC Space Ranger sample directory into an AnnData.

    Parameters
    ----------
    sample_dir
        Directory with ``filtered_feature_bc_matrix.h5`` and ``metadata.tsv``.
    label_col
        Metadata column holding the manual layer annotation (default ``"layer_guess"``;
        ``"layer_guess_reordered_short"`` gives the L1..L6/WM short form).
    coord_cols
        Metadata columns used for ``obsm["spatial"]`` (pixel coordinates by default).
    filter_to_labeled
        Drop spots whose layer label is missing (NaN) when True.

    Returns
    -------
    AnnData with raw counts in ``X``, ``obs["layer"]`` (ground truth), array coords in
    ``obs["array_row"/"array_col"]``, ``obsm["spatial"]``, and ``uns["sample"]``.
    """
    import scanpy as sc

    sample_dir = Path(sample_dir)
    h5 = sample_dir / "filtered_feature_bc_matrix.h5"
    if not h5.exists():
        raise FileNotFoundError(
            f"{h5} not found — run download_dlpfc({sample_dir.parent!r}) or fetch the sample."
        )
    import pandas as pd

    adata = sc.read_10x_h5(h5)
    adata.var_names_make_unique()

    meta = pd.read_csv(sample_dir / "metadata.tsv", sep="\t", index_col=0)
    common = adata.obs_names.intersection(meta.index)
    if len(common) == 0:
        raise ValueError("no barcodes shared between the count matrix and metadata.tsv")
    adata = adata[common].copy()
    meta = meta.loc[common]

    if label_col not in meta.columns:
        raise KeyError(f"label column {label_col!r} not in metadata columns {list(meta.columns)}")
    adata.obs["layer"] = meta[label_col].astype("category")
    missing = [c for c in coord_cols if c not in meta.columns]
    if missing:
        raise KeyError(f"coordinate columns {missing} not in metadata")
    adata.obsm["spatial"] = meta[list(coord_cols)].to_numpy(dtype=float)
    if {"row", "col"}.issubset(meta.columns):
        adata.obs["array_row"] = meta["row"].to_numpy()
        adata.obs["array_col"] = meta["col"].to_numpy()
    has_name = "sample_name" in meta
    adata.uns["sample"] = str(meta["sample_name"].iloc[0]) if has_name else sample_dir.name

    if filter_to_labeled:
        adata = adata[adata.obs["layer"].notna()].copy()
    return adata
