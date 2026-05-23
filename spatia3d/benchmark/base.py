"""Unified method interfaces for the Spatia3D benchmark (PLAN.md 2.2).

Every competitor and every Spatia3D component is wrapped as a ``Method`` exposing a single ``run``,
so the harness can score them identically. Adapters whose backing package/env is unavailable raise
``MethodUnavailable`` and the runner records a graceful skip — heavy or dependency-conflicting
baselines (cell2location, RCTD, STitch3D, ...) are meant to run in an isolated env behind the same
interface (see ``IsolatedEnvMethod``).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike


class MethodUnavailable(RuntimeError):
    """Raised by an adapter when its backing package or environment is not installed."""


class Method:
    """Base class: a named, categorised wrapper around one algorithm."""

    name: str = "method"
    task: str = "generic"  # "deconvolution" | "alignment" | "domain"
    is_ours: bool = False  # True for Spatia3D components, for highlighting in tables

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        tag = " (ours)" if self.is_ours else ""
        return f"<{self.task} method: {self.name}{tag}>"


class DeconvolutionMethod(Method, ABC):
    task = "deconvolution"

    @abstractmethod
    def run(
        self, Y: ArrayLike, V: ArrayLike, *, coords: ArrayLike | None = None, **kw
    ) -> np.ndarray:
        """Return cell-type proportions ``(n_spots, n_celltypes)`` (rows sum to 1)."""


class AlignmentMethod(Method, ABC):
    task = "alignment"

    @abstractmethod
    def run(
        self, coords_list: list[ArrayLike], *, features: list[ArrayLike] | None = None, **kw
    ) -> list[np.ndarray]:
        """Return aligned coordinates per slice (mapped into a common frame)."""


class DomainMethod(Method, ABC):
    task = "domain"

    @abstractmethod
    def run(self, X: ArrayLike, coords: ArrayLike, *, n_clusters: int, **kw) -> np.ndarray:
        """Return integer spatial-domain labels per spot ``(n_spots,)``."""


class IsolatedEnvMethod(Method):
    """Base for competitors that must run in a separate conda env via subprocess.

    Concrete subclasses set ``env_name``/``script`` and implement ``run`` by serialising inputs to a
    temp dir, invoking ``conda run -n <env> python <script>``, and reading back the prediction. This
    keeps conflicting dependency stacks (e.g. scvi-tools, R/spacexr) out of the main environment.
    """

    env_name: str = ""

    def _require_env(self) -> None:
        import shutil
        import subprocess

        if not shutil.which("conda"):
            raise MethodUnavailable("conda not found; cannot run isolated-env baseline")
        envs = subprocess.run(["conda", "env", "list"], capture_output=True, text=True).stdout
        if self.env_name and self.env_name not in envs:
            raise MethodUnavailable(
                f"conda env {self.env_name!r} not found — create it per docs/BENCHMARK.md"
            )
