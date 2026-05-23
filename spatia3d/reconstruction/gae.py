"""Spatial graph autoencoder for domain-embedding (C4, baseline form).

A symmetric GCN autoencoder over the spatial neighbour graph: the encoder propagates expression
across neighbours to a low-dimensional embedding, the decoder reconstructs expression from it, and
training minimises reconstruction error. The learned embedding is spatially smoothed *and*
expression-faithful, so clustering it recovers spatial domains far better than raw expression
(the STAGATE / SEDR family). This is the non-equivariant baseline form of C4; ``egnn.py`` is the
E(n)-equivariant upgrade (frame-invariant embedding, edge-list scalable).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from numpy.typing import ArrayLike
from sklearn.neighbors import kneighbors_graph
from torch import nn

__all__ = [
    "normalized_adjacency",
    "SpatialGAE",
    "train_gae",
    "spatial_gae_embedding",
    "refine_labels",
]


def refine_labels(labels: ArrayLike, coords: ArrayLike, *, k: int = 6, n_iter: int = 1):
    """Majority-vote spatial smoothing of cluster labels (SpaGCN/STAGATE-style refinement).

    Each spot is reassigned to the majority label among itself and its ``k`` nearest spatial
    neighbours, repeated ``n_iter`` times. Cleans up isolated misassignments and usually lifts ARI.
    """
    from sklearn.neighbors import NearestNeighbors

    labels = np.asarray(labels).copy()
    coords = np.asarray(coords, dtype=float)
    nn = NearestNeighbors(n_neighbors=min(k + 1, coords.shape[0])).fit(coords)
    idx = nn.kneighbors(coords, return_distance=False)  # includes self -> gentle smoothing
    for _ in range(n_iter):
        nb = labels[idx]
        new = np.empty_like(labels)
        for i, row in enumerate(nb):
            vals, counts = np.unique(row, return_counts=True)
            new[i] = vals[counts.argmax()]
        labels = new
    return labels


def normalized_adjacency(
    coords: ArrayLike, *, k: int = 6, device: str = "cpu"
) -> torch.Tensor:
    """Symmetric-normalised adjacency ``D^{-1/2}(A+I)D^{-1/2}`` of a k-NN spot graph, as a sparse
    torch tensor for message passing."""
    coords = np.asarray(coords, dtype=float)
    n = coords.shape[0]
    k = min(k, n - 1)
    W = kneighbors_graph(coords, n_neighbors=k, mode="connectivity", include_self=False)
    W = W.maximum(W.T)
    A = (W + sp.identity(n)).tocoo()  # add self-loops
    deg = np.asarray(A.sum(axis=1)).ravel()
    d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D = sp.diags(d_inv_sqrt)
    An = (D @ A @ D).tocoo()
    idx = torch.tensor(np.vstack([An.row, An.col]), dtype=torch.long)
    val = torch.tensor(An.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, val, (n, n)).coalesce().to(device)


class _GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, H: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        return torch.sparse.mm(A, self.lin(H))


class SpatialGAE(nn.Module):
    """Two-layer GCN encoder + symmetric GCN decoder."""

    def __init__(self, in_dim: int, hidden: int = 512, latent: int = 30) -> None:
        super().__init__()
        self.enc1 = _GCNLayer(in_dim, hidden)
        self.enc2 = _GCNLayer(hidden, latent)
        self.dec1 = _GCNLayer(latent, hidden)
        self.dec2 = _GCNLayer(hidden, in_dim)

    def encode(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        return self.enc2(F.relu(self.enc1(X, A)), A)

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        Z = self.encode(X, A)
        X_hat = self.dec2(F.relu(self.dec1(Z, A)), A)
        return Z, X_hat


def train_gae(
    X: ArrayLike,
    A: torch.Tensor,
    *,
    hidden: int = 512,
    latent: int = 30,
    epochs: int = 600,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str | None = None,
    seed: int = 0,
    verbose: bool = False,
) -> tuple[np.ndarray, list[float]]:
    """Train the GAE to reconstruct ``X`` over graph ``A``; return ``(embedding, loss_history)``.

    Uses CUDA when available (override with ``device``); moves ``A`` onto the same device.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    Xt = torch.as_tensor(np.asarray(X, dtype=np.float32), device=device)
    A = A.to(device)
    model = SpatialGAE(Xt.shape[1], hidden=hidden, latent=latent).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: list[float] = []
    model.train()
    for epoch in range(epochs):
        opt.zero_grad()
        _, X_hat = model(Xt, A)
        loss = F.mse_loss(X_hat, Xt)
        loss.backward()
        opt.step()
        history.append(float(loss.item()))
        if verbose and epoch % 100 == 0:
            print(f"epoch {epoch:4d}  recon_mse {loss.item():.4f}")
    model.eval()
    with torch.no_grad():
        Z = model.encode(Xt, A).cpu().numpy()
    return Z, history


def spatial_gae_embedding(
    X: ArrayLike, coords: ArrayLike, *, k: int = 6, device: str | None = None, **train_kwargs
) -> tuple[np.ndarray, list[float]]:
    """Build the spatial graph and train the GAE in one call -> ``(embedding, loss_history)``.

    Uses CUDA when available (override with ``device``).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    A = normalized_adjacency(coords, k=k, device=device)
    return train_gae(X, A, device=device, **train_kwargs)
