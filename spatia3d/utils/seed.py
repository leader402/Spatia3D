"""Global seeding for reproducible experiments."""

from __future__ import annotations

import os
import random

import numpy as np

__all__ = ["set_seed"]


def set_seed(seed: int = 0, *, deterministic_torch: bool = True) -> int:
    """Seed Python, NumPy and (if installed) PyTorch; return the seed used.

    With ``deterministic_torch`` the cuDNN backend is put in deterministic mode. Torch is imported
    lazily so the core package has no hard dependency on it.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:  # pragma: no cover - torch is an optional dependency
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    return seed
