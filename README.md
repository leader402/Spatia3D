# Spatia3D

Unified **slice registration + cell-type deconvolution + 3D spatial-domain reconstruction** for
spatial transcriptomics — solved as a single differentiable framework where the three tasks act as
priors for one another.

## Install

```bash
conda env create -f environment.yml
conda activate spatia3d
pip install -e .
```

(PyTorch is installed per machine for the right CUDA build.)

## Package layout

```
spatia3d/
  registration/    rigid (Kabsch / ICP) + differentiable optimal-transport slice alignment
  deconvolution/   ADMM cell-type deconvolution (Elastic Net + spatial total-variation)
  reconstruction/  spatial graph autoencoders (GCN / GAT) + domain clustering
  joint/           joint deconvolution<->domain optimization; end-to-end differentiable model
  datasets/        ground-truth simulators + data loaders
  metrics/         domain (ARI/NMI/ASW), deconvolution (RMSE/JSD/PCC), alignment metrics
  benchmark/       unified method-comparison harness
  pipeline.py      run_unified: align -> 3D graph -> joint deconvolution<->domain
```

## Quickstart

```python
from spatia3d.datasets import simulate_3d
from spatia3d.pipeline import run_unified

sim = simulate_3d(n_slices=4, spots_per_side=12, seed=0)
res = run_unified(
    [s.coords_obs for s in sim.slices],
    [s.expression for s in sim.slices],
    sim.signatures,
    n_domains=sim.meta["n_domains"],
)
res.proportions   # cell-type proportions per spot
res.domains       # 3D spatial-domain labels
```
