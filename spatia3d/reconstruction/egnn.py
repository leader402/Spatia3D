"""C4 architecture upgrade — E(n)-equivariant spatial graph autoencoder (EGNN-style, torch/GPU).

The GCN (``gae.py``) and GAT (``gat.py``) C4 forms treat the 3D coordinates as ordinary features, so
their embedding depends on the *arbitrary* global frame the aligned stack happens to sit in: rotate
the whole reconstruction and the domains can move. This module processes geometry **equivariantly**
(Satorras, Hoogeboom & Welling, EGNN 2021): every message depends only on pairwise *squared
distances*, so the node embedding is E(3)-**invariant** — rotating or translating the whole 3D stack
leaves every spot's domain embedding unchanged — while coordinates update equivariantly. For 3D
spatial transcriptomics, where the global orientation carries no biology, this is the correct
inductive bias and removes the frame dependence the GCN/GAT embeddings inherit.

Messages run over a k-NN **edge list** (``O(N·k)``), not the dense ``O(N²)`` attention matrix of
``gat.py``, so the same model scales from the simulator to Stereo-seq counts. Scatter aggregation
uses ``index_add_`` (built into torch — no compiled ``torch_scatter`` dependency). The bottleneck
node features are the (invariant) domain embedding; cluster them as with the other C4 encoders.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import ArrayLike
from sklearn.neighbors import kneighbors_graph
from torch import nn

__all__ = ["knn_edge_index", "EGNNLayer", "EquivariantGAE", "train_egnn", "spatial_egnn_embedding"]


def knn_edge_index(coords: ArrayLike, *, k: int = 6, device: str = "cpu") -> torch.Tensor:
    """Symmetric k-NN graph as a directed ``(2, E)`` edge list ``[src; dst]`` (no self-loops).

    Both directions of every neighbour pair are present (the graph is symmetrised) — this is what
    the equivariant message passing aggregates over.
    """
    coords = np.asarray(coords, dtype=float)
    n = coords.shape[0]
    k = min(k, n - 1)
    W = kneighbors_graph(coords, n_neighbors=k, mode="connectivity", include_self=False)
    W = W.maximum(W.T).tocoo()  # symmetric -> both (i,j) and (j,i) present
    return torch.tensor(np.vstack([W.row, W.col]), dtype=torch.long, device=device)


def _scatter_sum(src: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    """Sum rows of ``src`` into ``n`` buckets given by ``index`` (torch-native scatter)."""
    out = src.new_zeros(n, src.shape[1])
    out.index_add_(0, index, src)
    return out


class EGNNLayer(nn.Module):
    """One E(n)-equivariant graph-conv layer: invariant feature update + equivariant coord step."""

    def __init__(self, hidden: int, *, coord_update: bool = True) -> None:
        super().__init__()
        self.coord_update = coord_update
        # Edge message from (h_i, h_j, ||x_i - x_j||²) — distance makes it E(3)-invariant.
        self.phi_e = nn.Sequential(
            nn.Linear(2 * hidden + 1, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU()
        )
        # Per-edge scalar weighting the (equivariant) coordinate difference.
        self.phi_x = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        # Invariant node update from aggregated messages.
        self.phi_h = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )

    def forward(self, h, x, edge_index):
        src, dst = edge_index[0], edge_index[1]
        diff = x[dst] - x[src]  # (E, dim) — transforms equivariantly under rotation
        dist2 = (diff**2).sum(-1, keepdim=True)  # (E, 1) — invariant
        m = self.phi_e(torch.cat([h[dst], h[src], dist2], dim=-1))  # (E, hidden)
        n = h.shape[0]
        if self.coord_update:
            # Equivariant: weighted sum of normalised coordinate differences, mean over neighbours.
            w = diff / (dist2.sqrt() + 1.0) * self.phi_x(m)  # (E, dim)
            deg = _scatter_sum(torch.ones_like(dist2), dst, n).clamp_min(1.0)
            x = x + _scatter_sum(w, dst, n) / deg
        agg = _scatter_sum(m, dst, n)  # (N, hidden) — invariant aggregate
        h = h + self.phi_h(torch.cat([h, agg], dim=-1))
        return h, x


class EquivariantGAE(nn.Module):
    """E(n)-equivariant graph autoencoder; ``encode`` gives the E(3)-invariant domain embedding."""

    def __init__(self, in_dim: int, hidden: int = 64, latent: int = 30, n_layers: int = 3) -> None:
        super().__init__()
        self.embed = nn.Linear(in_dim, hidden)
        self.layers = nn.ModuleList(EGNNLayer(hidden) for _ in range(n_layers))
        self.to_latent = nn.Linear(hidden, latent)
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden), nn.SiLU(), nn.Linear(hidden, in_dim)
        )

    def encode(self, X: torch.Tensor, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.embed(X)
        for layer in self.layers:
            h, x = layer(h, x, edge_index)
        return self.to_latent(h)  # invariant features only -> embedding is E(3)-invariant

    def forward(self, X, x, edge_index) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(X, x, edge_index)
        return z, self.decoder(z)


def train_egnn(
    X: ArrayLike,
    coords: ArrayLike,
    *,
    k: int = 6,
    hidden: int = 64,
    latent: int = 30,
    n_layers: int = 3,
    epochs: int = 600,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, list[float]]:
    """Train the equivariant graph autoencoder on ``X`` over the k-NN graph of ``coords``.

    ``coords`` may be 2-D (in-plane) or 3-D (the aligned stack). Returns ``(embedding,
    loss_history)``; the embedding is E(3)-invariant. CUDA when available (override ``device``).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    Xt = torch.as_tensor(np.asarray(X, dtype=np.float32), device=device)
    xt = torch.as_tensor(np.asarray(coords, dtype=np.float32), device=device)
    edge_index = knn_edge_index(coords, k=k, device=device)
    model = EquivariantGAE(Xt.shape[1], hidden=hidden, latent=latent, n_layers=n_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: list[float] = []
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        _, X_hat = model(Xt, xt, edge_index)
        loss = F.mse_loss(X_hat, Xt)
        loss.backward()
        opt.step()
        history.append(float(loss.item()))
    model.eval()
    with torch.no_grad():
        z = model.encode(Xt, xt, edge_index).cpu().numpy()
    return z, history


def spatial_egnn_embedding(
    X: ArrayLike, coords: ArrayLike, *, k: int = 6, device: str | None = None, **train_kwargs
) -> tuple[np.ndarray, list[float]]:
    """Train the equivariant GAE in one call -> ``(embedding, loss_history)``."""
    return train_egnn(X, coords, k=k, device=device, **train_kwargs)
