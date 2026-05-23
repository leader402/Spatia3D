"""Alignment method adapters for the benchmark.

In-env: Spatia3D's pivot rigid alignment (paired Kabsch and ICP). Real external competitor: PASTE
(OT-based, Nat Methods 2022), run in its own conda env via subprocess because its OT line_search
conflicts with the numpy-2 POT the main env needs. STitch3D / GPSA / CAST plug in the same way
(see docs/BENCHMARK.md).
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from spatia3d.benchmark.base import AlignmentMethod, IsolatedEnvMethod, MethodUnavailable

__all__ = ["PivotRigid", "PivotICP", "Paste", "ALIGNMENT_METHODS"]

_PASTE_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "baselines" / "run_paste.py"


class PivotRigid(AlignmentMethod):
    """Spatia3D pivot registration with closed-form Kabsch (requires index correspondence)."""

    name = "Spatia3D-pivot"
    is_ours = True

    def run(self, coords_list, *, features=None, **kw):
        from spatia3d.registration import pivot_register

        aligned, _ = pivot_register([np.asarray(c, float) for c in coords_list], method="paired")
        return aligned


class PivotICP(AlignmentMethod):
    """Spatia3D pivot registration via ICP (unpaired / cross-platform)."""

    name = "Spatia3D-ICP"
    is_ours = True

    def run(self, coords_list, *, features=None, **kw):
        from spatia3d.registration import pivot_register

        aligned, _ = pivot_register(
            [np.asarray(c, float) for c in coords_list], method="icp", max_iter=200
        )
        return aligned


class Paste(IsolatedEnvMethod, AlignmentMethod):
    """PASTE (Zeira et al., Nat Methods 2022): OT pairwise slice alignment.

    Runs in the isolated ``spatia3d-paste`` conda env via subprocess (inputs serialised to a temp
    dir, ``conda run`` the runner script, read aligned coords back).
    """

    name = "PASTE"
    env_name = "spatia3d-paste"

    def run(self, coords_list, *, features: list[ArrayLike] | None = None, **kw):
        self._require_env()
        if features is None:
            raise ValueError("PASTE needs per-slice expression features")
        if not _PASTE_SCRIPT.exists():
            raise MethodUnavailable(f"PASTE runner script missing: {_PASTE_SCRIPT}")
        n = len(coords_list)
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            for i, (c, X) in enumerate(zip(coords_list, features, strict=True)):
                np.save(work / f"coords_{i}.npy", np.asarray(c, dtype=float))
                np.save(work / f"feat_{i}.npy", np.asarray(X, dtype=float))
            (work / "meta.json").write_text(json.dumps({"n_slices": n}))
            proc = subprocess.run(
                ["conda", "run", "-n", self.env_name, "python", str(_PASTE_SCRIPT), str(work)],
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                raise MethodUnavailable(f"PASTE subprocess failed: {proc.stderr.strip()[-200:]}")
            return [np.load(work / f"aligned_{i}.npy") for i in range(n)]


def ALIGNMENT_METHODS(*, include_ours: bool = True, include_paste: bool = True):
    """Default alignment roster for the benchmark."""
    methods: list[AlignmentMethod] = []
    if include_ours:
        methods += [PivotRigid(), PivotICP()]
    if include_paste:
        methods.append(Paste())
    return methods
