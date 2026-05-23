"""STitch3D adapter — the head competitor (Wang et al., Nat Mach Intell 2023).

STitch3D jointly does alignment + deconvolution + 3D domains, so unlike the single-slice adapters it
takes *multiple* slices (raw counts + coords) and a *single-cell reference*, and returns proportions
+ a latent embedding (clustered into domains). It runs in the isolated `spatia3d-stitch3d` env
(2022 stack) on CPU via subprocess — its torch (cu117) predates the Blackwell GPU.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from spatia3d.benchmark.base import IsolatedEnvMethod, MethodUnavailable

__all__ = ["Stitch3D"]

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "baselines" / "run_stitch3d.py"


class Stitch3D(IsolatedEnvMethod):
    """Joint alignment + deconvolution + 3D domain method; custom multi-slice run signature."""

    name = "STitch3D"
    task = "joint"
    env_name = "spatia3d-stitch3d"

    def run(
        self,
        coords_list: list[ArrayLike],
        counts_list: list[ArrayLike],
        ref_counts: ArrayLike,
        ref_labels: ArrayLike,
        *,
        array_list: list[ArrayLike] | None = None,
        training_steps: int = 4000,
        n_hvg_group: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Return (proportions [spots, n_celltypes], latent [spots, d], celltype-order list).

        ``array_list`` gives integer Visium array indices per slice (STitch3D's ICP uses them to
        gauge neighbour density); defaults to ``coords_list``.
        """
        self._require_env()
        if not _SCRIPT.exists():
            raise MethodUnavailable(f"STitch3D runner missing: {_SCRIPT}")
        arrays = array_list if array_list is not None else coords_list
        n = len(coords_list)
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            for i, (c, x, a) in enumerate(zip(coords_list, counts_list, arrays, strict=True)):
                np.save(work / f"coords_{i}.npy", np.asarray(c, dtype=float))
                np.save(work / f"counts_{i}.npy", np.asarray(x, dtype=float))
                np.save(work / f"array_{i}.npy", np.asarray(a, dtype=float))
            np.save(work / "ref_counts.npy", np.asarray(ref_counts, dtype=float))
            np.save(work / "ref_labels.npy", np.asarray(ref_labels))
            meta = {"n_slices": n, "training_steps": int(training_steps)}
            if n_hvg_group:
                meta["n_hvg_group"] = int(n_hvg_group)
            (work / "meta.json").write_text(json.dumps(meta))
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="")  # old torch can't use Blackwell -> CPU
            proc = subprocess.run(
                ["conda", "run", "-n", self.env_name, "python", str(_SCRIPT), str(work)],
                capture_output=True,
                text=True,
                env=env,
            )
            if proc.returncode != 0:
                raise MethodUnavailable(f"STitch3D subprocess failed: {proc.stderr.strip()[-300:]}")
            props = np.load(work / "proportions.npy")
            latent = np.load(work / "latent.npy")
            celltypes = json.loads((work / "celltypes.json").read_text())
            return props, latent, celltypes
