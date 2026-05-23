"""Drosophila 3D Stereo-seq loader (Flysta3D; Wang et al., Dev Cell 2022) — real serial-section 3D.

A genuine 3D-reconstruction benchmark: each developmental stage is a stack of serial Stereo-seq
sections of one specimen, with (a) per-section raw coordinates in independent frames (the
registration *input*), (b) the paper's 3D-reconstructed coordinates ``new_x/new_y/new_z`` (a
registration *ground truth*), and (c) an anatomical ``annotation`` (CNS, muscle, fat body, ...) — a
genuinely spatial domain label, unlike the heart's transcriptomic clustering. New platform
(Stereo-seq) + new tissue for the 3D story.

``download_flysta3d`` fetches a stage's processed ``.h5ad`` from the BGI mirror; ``load_flysta3d``
splits it into per-section AnnData (ordered by depth) with the raw coords in ``obsm["spatial"]`` (to
register) and the reference 3D coords in ``obsm["spatial_ref"]``.
"""

from __future__ import annotations

from pathlib import Path
from urllib.request import Request, urlopen

__all__ = ["download_flysta3d", "load_flysta3d", "FLYSTA3D_STAGES"]

#: Developmental stages available from the mirror (embryo E*, larva L*).
FLYSTA3D_STAGES = ("E14-16h", "E16-18h", "L1", "L2", "L3")
_BASE = "https://www.bgiocean.com/vt3d_example/download/flysta3d"


def download_flysta3d(stage: str = "E16-18h", dest_root: str | Path = "data/flysta3d") -> Path:
    """Download one stage's processed ``.h5ad`` (3D coords + annotation) from the BGI mirror.

    Returns the file path; existing files are not re-downloaded. ~45 MB for E16-18h.
    """
    if stage not in FLYSTA3D_STAGES:
        raise ValueError(f"stage {stage!r} not in {FLYSTA3D_STAGES}")
    dest = Path(dest_root)
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / f"{stage}.h5ad"
    if not out.exists():
        url = f"{_BASE}/{stage}_a_count_normal_stereoseq.h5ad"
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=600) as r, open(out, "wb") as f:  # noqa: S310 (trusted mirror)
            import shutil

            shutil.copyfileobj(r, f)
    return out


def load_flysta3d(
    stage: str = "E16-18h", data_dir: str | Path = "data/flysta3d", *, download: bool = True
):
    """Load a Flysta3D stage as a list of per-section AnnData (ordered by reconstructed depth).

    Each section carries counts in ``X``; ``obsm["spatial"]`` the raw per-section coordinates (own
    frame — what registration must align); ``obsm["spatial_ref"]`` the paper's reconstructed
    ``(new_x, new_y, new_z)`` (registration ground truth); ``obs["domain"]`` the anatomical
    annotation; ``uns["sample"]`` the slice id. Feed the raw coords to ``differentiable_ot_align`` /
    ``run_unified``; score against the annotation or the reference coords.
    """
    import anndata as ad

    path = Path(data_dir) / f"{stage}.h5ad"
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"{path} not found — run download_flysta3d({stage!r})")
        download_flysta3d(stage, data_dir)

    adata = ad.read_h5ad(path)
    adata.var_names_make_unique()
    # order sections by reconstructed depth (new_z)
    z_of = {s: adata.obs.loc[adata.obs["slice_ID"] == s, "new_z"].iloc[0]
            for s in adata.obs["slice_ID"].unique()}
    sections = []
    for sid in sorted(z_of, key=lambda s: z_of[s]):
        sub = adata[adata.obs["slice_ID"] == sid].copy()
        sub.obsm["spatial"] = sub.obs[["raw_x", "raw_y"]].to_numpy(dtype=float)  # un-aligned input
        sub.obsm["spatial_ref"] = sub.obs[["new_x", "new_y", "new_z"]].to_numpy(dtype=float)  # GT
        sub.obs["domain"] = sub.obs["annotation"].astype(str).to_numpy()
        sub.uns["sample"] = str(sid)
        sections.append(sub)
    return sections
