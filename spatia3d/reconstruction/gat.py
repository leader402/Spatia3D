"""C4 upgrade — spatial graph *attention* autoencoder (STAGATE/OptiGraph3D family, torch/GPU).

Replaces the plain GCN message passing of ``gae.py`` with multi-head graph attention: the model
learns per-edge weights instead of using the fixed normalised adjacency, then reconstructs (raw HVG)
expression from the attended neighbourhood. The bottleneck embedding clusters into spatial domains.
This is the non-equivariant attention form of C4 (the OptiGraph3D multi-head-GAT idea, modernised);
the SE(3)-equivariant 3D-transformer + class-token is the further upgrade.

Attention is computed densely with an adjacency mask (the Velickovic factorisation
``e_ij = LeakyReLU(s_i + t_j)`` avoids materialising per-edge feature pairs). This suits
simulator/DLPFC scale (a few thousand spots on GPU); for Stereo-seq millions use a sparse
edge-indexed GAT (TODO).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from numpy.typing import ArrayLike
from sklearn.neighbors import kneighbors_graph
from torch import nn

__all__ = ["adjacency_mask", "GATLayer", "SpatialGAT", "train_gat"]


def adjacency_mask(coords: ArrayLike, *, k: int = 6, device: str = "cpu") -> torch.Tensor:
    """Dense boolean k-NN adjacency (symmetric, with self-loops) for masked attention."""
    coords = np.asarray(coords, dtype=float)
    n = coords.shape[0]
    k = min(k, n - 1)
    W = kneighbors_graph(coords, n_neighbors=k, mode="connectivity", include_self=False)
    W = (W.maximum(W.T) + sp.identity(n)).astype(bool)
    return torch.tensor(np.asarray(W.todense()), dtype=torch.bool, device=device)


class GATLayer(nn.Module):
    """Multi-head graph attention over a dense adjacency mask."""

    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, *, concat: bool = True) -> None:
        super().__init__()
        self.heads, self.out_dim, self.concat = heads, out_dim, concat
        self.W = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.a_src = nn.Parameter(torch.empty(heads, out_dim))
        self.a_dst = nn.Parameter(torch.empty(heads, out_dim))
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        n = h.shape[0]
        Wh = self.W(h).view(n, self.heads, self.out_dim)  # (n, H, d)
        s = (Wh * self.a_src).sum(-1)  # (n, H)
        t = (Wh * self.a_dst).sum(-1)  # (n, H)
        e = F.leaky_relu(s.unsqueeze(1) + t.unsqueeze(0), 0.2)  # (n, n, H): i->j scores
        e = e.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        alpha = torch.softmax(e, dim=1)  # over neighbours j
        out = torch.einsum("ijh,jhd->ihd", alpha, Wh)  # attended aggregation
        return out.reshape(n, self.heads * self.out_dim) if self.concat else out.mean(1)


class SpatialGAT(nn.Module):
    """Symmetric graph-attention autoencoder; ``encode`` returns the domain embedding."""

    def __init__(self, in_dim: int, hidden: int = 64, latent: int = 30, heads: int = 4) -> None:
        super().__init__()
        self.enc1 = GATLayer(in_dim, hidden, heads, concat=True)
        self.enc2 = GATLayer(hidden * heads, latent, 1, concat=False)
        self.dec1 = GATLayer(latent, hidden, heads, concat=True)
        self.dec2 = GATLayer(hidden * heads, in_dim, 1, concat=False)

    def encode(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.enc2(F.elu(self.enc1(x, mask)), mask)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x, mask)
        x_hat = self.dec2(F.elu(self.dec1(z, mask)), mask)
        return z, x_hat


def train_gat(
    X: ArrayLike,
    coords: ArrayLike,
    *,
    k: int = 6,
    hidden: int = 64,
    latent: int = 30,
    heads: int = 4,
    epochs: int = 600,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, list[float]]:
    """Train the graph-attention autoencoder on ``X``; return (embedding, loss_history)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    Xt = torch.as_tensor(np.asarray(X, dtype=np.float32), device=device)
    mask = adjacency_mask(coords, k=k, device=device)
    model = SpatialGAT(Xt.shape[1], hidden=hidden, latent=latent, heads=heads).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: list[float] = []
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        _, x_hat = model(Xt, mask)
        loss = F.mse_loss(x_hat, Xt)
        loss.backward()
        opt.step()
        history.append(float(loss.item()))
    model.eval()
    with torch.no_grad():
        z = model.encode(Xt, mask).cpu().numpy()
    return z, history
