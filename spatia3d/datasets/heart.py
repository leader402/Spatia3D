"""Developing human heart loader (Asp et al., Cell 2019) — real 3D multi-slice ST benchmark.

The original-paper second dataset and the field's reference 3D developmental atlas: spatial
transcriptomics over serial sections of human embryonic hearts at three stages (4.5–5, 6.5, 9
post-conception weeks). Each section is an independent capture, so its spot coordinates live in
their own frame — exactly the setting Spatia3D's registration aligns into a 3D volume. ``res.0.8``
(the paper's Seurat clustering, 10 domains) serves as a domain reference.

``download_heart`` fetches the filtered ST count matrix + meta (and, optionally, the scRNA reference
for deconvolution) from Mendeley Data (doi 10.17632/mbvhhf8m62). ``load_heart`` returns one AnnData
per serial section of a chosen stage — the 9 PCW heart (``weeks=9``) is 6 clean serial sections, the
best 3D stack — ready for :func:`spatia3d.datasets.load_spatial_stack` / ``run_unified``.
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

__all__ = ["download_heart", "load_heart", "load_heart_reference", "HEART_STAGES"]


def _fetch(url: str, dest: Path) -> None:
    """Download ``url`` to ``dest`` following redirects with a browser UA (Mendeley needs both)."""
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=300) as r, open(dest, "wb") as f:  # noqa: S310 (trusted Mendeley URL)
        shutil.copyfileobj(r, f)

#: Spots-per-stage / serial-section structure of the filtered ST data (post QC).
HEART_STAGES = {5: "4.5-5 PCW (FH5, 4 sections)", 6: "6.5 PCW (FH6, 9 sections)",
                9: "9 PCW (FH9, 6 sections)"}

# Mendeley Data direct file URLs (doi 10.17632/mbvhhf8m62 — count matrices + meta).
_ST_ZIP = ("https://data.mendeley.com/public-files/datasets/mbvhhf8m62/files/"
           "f76ec6ad-addd-41c3-9eec-56e31ddbac71/file_downloaded")
_SC_ZIP = ("https://data.mendeley.com/public-files/datasets/mbvhhf8m62/files/"
           "09bd6f4a-a8c4-410b-b9f2-16bc264be790/file_downloaded")
_ST_DIR = "filtered_ST_matrix_and_meta_data"
_SC_DIR = "Developmental_heart_filtered_scRNA-seq_and_meta_data"


def download_heart(dest_root: str | Path = "data/heart", *, scrna: bool = False) -> Path:
    """Download + unzip the filtered developing-heart ST data (and optionally the scRNA reference).

    Returns the ``dest_root`` path. Existing unzipped folders are not re-downloaded. With
    ``scrna=True`` also fetches the single-cell reference (for :func:`build_signatures`).
    """
    dest = Path(dest_root)
    dest.mkdir(parents=True, exist_ok=True)
    jobs = [("ST.zip", _ST_ZIP, _ST_DIR)]
    if scrna:
        jobs.append(("scRNA.zip", _SC_ZIP, _SC_DIR))
    for zip_name, url, out_dir in jobs:
        if (dest / out_dir).exists():
            continue
        zpath = dest / zip_name
        if not zpath.exists():
            _fetch(url, zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(dest)
    return dest


def load_heart(data_dir: str | Path = "data/heart", *, weeks: int = 9, download: bool = True):
    """Load the developing-heart ST data for one stage as a list of per-section AnnData.

    Parameters
    ----------
    data_dir
        Folder holding ``filtered_ST_matrix_and_meta_data/`` (auto-downloaded if ``download``).
    weeks
        Developmental stage / heart to load: ``5``, ``6``, or ``9`` (see :data:`HEART_STAGES`).
        ``9`` (the FH9 heart, 6 serial sections) is the cleanest 3D stack.
    download
        Fetch the data first if the folder is missing.

    Returns
    -------
    list[AnnData]
        One AnnData per serial section (ordered by section index = z-order): raw counts in ``X``,
        ``obsm["spatial"]`` the section's array coordinates (own frame), ``obs["domain"]`` the
        paper's ``res.0.8`` clustering, ``uns["sample"]`` the section id. Feed straight into
        :func:`spatia3d.datasets.load_spatial_stack`.
    """
    import anndata as ad
    import pandas as pd

    d = Path(data_dir) / _ST_DIR
    if not d.exists():
        if not download:
            raise FileNotFoundError(f"{d} not found — run download_heart({str(data_dir)!r})")
        download_heart(data_dir)

    meta = pd.read_csv(d / "meta_data.tsv.gz", sep="\t", index_col=0)
    if weeks not in set(meta["weeks"]):
        raise ValueError(f"weeks={weeks} not in data {sorted(set(meta['weeks']))}")
    meta = meta[meta["weeks"] == weeks].copy()
    meta["section"] = [int(s.split("x")[0]) for s in meta.index]  # leading id = z-order

    # Read only this stage's spot columns (not all 3111) as float32: the full 39740-gene dense
    # matrix balloons to ~1 GB in pandas, which has spiked WSL OOM under concurrent loads.
    import gzip

    with gzip.open(d / "filtered_matrix.tsv.gz", "rt") as f:
        header = f.readline().rstrip("\n").split("\t")
    wanted = set(meta.index)
    cols = [0] + [i for i, c in enumerate(header) if c in wanted]  # positional: gene col + spots
    mat = pd.read_csv(
        d / "filtered_matrix.tsv.gz", sep="\t", usecols=cols,
        dtype=dict.fromkeys([header[i] for i in cols[1:]], "float32"),
    )
    mat = mat.set_index(mat.columns[0])  # genes index; spots are the columns
    genes = mat.index.astype(str)

    adatas = []
    for sec in sorted(meta["section"].unique()):
        sub = meta[meta["section"] == sec]
        X = mat[sub.index].T.to_numpy(dtype="float32")  # spots x genes (this section)
        a = ad.AnnData(X)
        a.var_names = genes
        a.obs_names = list(sub.index)
        a.obsm["spatial"] = sub[["new_x", "new_y"]].to_numpy(float)
        a.obs["domain"] = sub["res.0.8"].astype(str).to_numpy()
        a.uns["sample"] = str(sub["Sample"].iloc[0])
        adatas.append(a)
    return adatas


def load_heart_reference(data_dir: str | Path = "data/heart", *, download: bool = True):
    """Load the developing-heart scRNA reference for deconvolution-signature estimation.

    Returns ``(counts, celltypes, gene_names)`` — ``counts`` is ``(n_cells, n_genes)`` and
    ``celltypes`` the per-cell annotation. Pass to ``build_signatures`` to get the signature
    matrix ``V`` for deconvolving the ST sections.

    Note: the reference uses gene **symbols** while the ST matrix uses **Ensembl IDs** — bridge the
    two namespaces (e.g. via an Ensembl↔symbol map) before intersecting genes for deconvolution.
    """
    import pandas as pd

    d = Path(data_dir) / _SC_DIR
    if not d.exists():
        if not download:
            raise FileNotFoundError(
                f"{d} not found — run download_heart({str(data_dir)!r}, scrna=True)"
            )
        download_heart(data_dir, scrna=True)

    meta = pd.read_csv(d / "all_cells_meta_data_filtered.tsv.gz", sep="\t", index_col=0)
    mat = pd.read_csv(  # genes x cells
        d / "all_cells_count_matrix_filtered.tsv.gz", sep="\t", index_col=0
    )
    cells = meta.index.intersection(mat.columns)
    counts = mat[cells].T.to_numpy(dtype="float32")  # cells x genes
    return counts, meta.loc[cells, "celltype"].to_numpy(), list(mat.index.astype(str))
