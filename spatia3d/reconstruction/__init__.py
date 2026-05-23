"""C4 — 3D-native graph learning for spatial-domain reconstruction.

A cross-slice 3D neighbour graph over the aligned stack; node features are expression, processed
with 3D geometry. Three encoder forms, increasing in inductive bias: a GCN autoencoder (baseline), a
graph-*attention* autoencoder (STAGATE/OptiGraph3D family), and the **E(n)-equivariant** autoencoder
(``egnn``) whose embedding is invariant to the arbitrary global frame of the 3D reconstruction and
which runs on a k-NN edge list (scales past the dense-attention form). Cluster any embedding with
``cluster_domains`` (optionally spatially ``refine``d).

Available (lazy, torch): GCN (normalized_adjacency, SpatialGAE, train_gae, spatial_gae_embedding,
refine_labels), graph-attention (adjacency_mask, GATLayer, SpatialGAT, train_gat), and equivariant
(knn_edge_index, EGNNLayer, EquivariantGAE, train_egnn, spatial_egnn_embedding).
"""

from spatia3d.reconstruction.clustering import cluster_domains as cluster_domains  # noqa: F401

_GAE = {"normalized_adjacency", "SpatialGAE", "train_gae", "spatial_gae_embedding", "refine_labels"}
_GAT = {"adjacency_mask", "GATLayer", "SpatialGAT", "train_gat"}
_EGNN = {"knn_edge_index", "EGNNLayer", "EquivariantGAE", "train_egnn", "spatial_egnn_embedding"}
__all__ = sorted(_GAE | _GAT | _EGNN | {"cluster_domains"})


def __getattr__(name: str):
    # Lazy import: only pull in torch-dependent symbols when actually requested.
    if name in _GAE:
        from spatia3d.reconstruction import gae

        return getattr(gae, name)
    if name in _GAT:
        from spatia3d.reconstruction import gat

        return getattr(gat, name)
    if name in _EGNN:
        from spatia3d.reconstruction import egnn

        return getattr(egnn, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
