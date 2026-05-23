"""Spatial neighbourhood graphs for the deconvolution TV regulariser."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike
from sklearn.neighbors import kneighbors_graph

__all__ = ["spatial_laplacian"]


def spatial_laplacian(coords: ArrayLike, *, k: int = 6, normalized: bool = False) -> sp.csr_matrix:
    """Symmetric graph Laplacian of a k-NN spot graph.

    The column-wise TV penalty ``tr(P.T L P)`` uses this ``L`` to encourage neighbouring spots to
    have similar cell-type proportions. With ``normalized`` the symmetric normalised Laplacian
    ``I - D^{-1/2} W D^{-1/2}`` is returned instead of the combinatorial ``D - W``.
    """
    coords = np.asarray(coords, dtype=float)
    n = coords.shape[0]
    k = min(k, n - 1)
    W = kneighbors_graph(coords, n_neighbors=k, mode="connectivity", include_self=False)
    W = W.maximum(W.T)  # make the adjacency symmetric
    deg = np.asarray(W.sum(axis=1)).ravel()
    if normalized:
        with np.errstate(divide="ignore"):
            d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
        D_inv_sqrt = sp.diags(d_inv_sqrt)
        return (sp.identity(n) - D_inv_sqrt @ W @ D_inv_sqrt).tocsr()
    return (sp.diags(deg) - W).tocsr()
