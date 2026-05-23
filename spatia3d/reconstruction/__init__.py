"""C4 — 3D-native graph learning for spatial-domain reconstruction.

True cross-slice 3D adjacency graph; node features include 3D coords + expression + learnable
class-token (keep OptiGraph3D's semi-supervised class-token idea). The SE(3)/translation-equivariant
or 3D-position-encoded graph transformer is the further upgrade.

Available (lazy, torch): GCN autoencoder (normalized_adjacency, SpatialGAE, train_gae,
spatial_gae_embedding, refine_labels) and the graph-*attention* autoencoder upgrade
(adjacency_mask, GATLayer, SpatialGAT, train_gat).
TODO(M5): build_3d_graph(), SE(3)-equivariant transformer, class_token semi-supervision.
"""

from spatia3d.reconstruction.clustering import cluster_domains as cluster_domains  # noqa: F401

_GAE = {"normalized_adjacency", "SpatialGAE", "train_gae", "spatial_gae_embedding", "refine_labels"}
_GAT = {"adjacency_mask", "GATLayer", "SpatialGAT", "train_gat"}
__all__ = sorted(_GAE | _GAT | {"cluster_domains"})


def __getattr__(name: str):
    # Lazy import: only pull in torch-dependent symbols when actually requested.
    if name in _GAE:
        from spatia3d.reconstruction import gae

        return getattr(gae, name)
    if name in _GAT:
        from spatia3d.reconstruction import gat

        return getattr(gat, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
