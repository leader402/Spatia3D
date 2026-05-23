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
  registration/    rigid (Kabsch / ICP) + differentiable optimal-transport alignment, rigid & non-rigid
  deconvolution/   ADMM deconvolution (Elastic Net + spatial-TV) + reference-signature estimation
                   + per-spot uncertainty (bootstrap / Laplace)
  reconstruction/  spatial graph autoencoders — GCN, GAT, and an E(n)-equivariant encoder — + clustering
  joint/           joint deconvolution<->domain optimization; end-to-end differentiable model
  datasets/        ground-truth simulators + DLPFC and a platform-agnostic multi-slice loader
  metrics/         domain (ARI/NMI/ASW), deconvolution (RMSE/JSD/PCC, calibration), alignment metrics
  benchmark/       unified method-comparison harness
  pipeline.py      run_unified: align -> 3D graph -> joint deconvolution<->domain (or end-to-end)
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

`joint="end_to_end"` co-trains registration with deconvolution and domains in one gradient flow;
`align`/`joint` switches make the unified-vs-staged ablation a single entry point.

### Deconvolution with uncertainty, from a single-cell reference

```python
from spatia3d.deconvolution import build_signatures, deconvolve_admm

V, names = build_signatures(ref.counts, ref.labels)        # estimate signatures from scRNA reference
res = deconvolve_admm(Y, V, coords=coords, return_uncertainty=True)
res.proportions, res.proportions_sd                        # estimate + per-spot, per-cell-type SD
```

### Real data, any platform → pipeline inputs

```python
from spatia3d.datasets import load_spatial_stack

stack = load_spatial_stack(["s1.h5ad", "s2.h5ad", "s3.h5ad"], n_top_genes=2000)
run_unified(*stack.pipeline_inputs(), V, n_domains=7)      # Visium / Stereo-seq / MERFISH / ...
```

### Frame-invariant 3D domain embedding (E(n)-equivariant)

```python
from spatia3d.reconstruction import spatial_egnn_embedding, cluster_domains

z, _ = spatial_egnn_embedding(expression, coords_3d)       # embedding invariant to the global frame
domains = cluster_domains(z, n_clusters=7, coords=coords_3d, refine=True)
```
