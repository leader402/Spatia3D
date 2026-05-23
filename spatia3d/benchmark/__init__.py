"""Unified benchmark harness (PLAN.md 2.2): wrap every competitor and every Spatia3D component
behind one interface, score them identically, tabulate.

Interfaces: DeconvolutionMethod / AlignmentMethod / DomainMethod (+ IsolatedEnvMethod for
dependency-conflicting baselines).  Runners: benchmark_deconvolution / _alignment / _domain.
Deconvolution adapters live in `benchmark.deconvolution`; see docs/BENCHMARK.md for the roster.
"""
from spatia3d.benchmark.alignment import ALIGNMENT_METHODS
from spatia3d.benchmark.base import (
    AlignmentMethod,
    DeconvolutionMethod,
    DomainMethod,
    IsolatedEnvMethod,
    Method,
    MethodUnavailable,
)
from spatia3d.benchmark.deconvolution import DECONVOLUTION_METHODS
from spatia3d.benchmark.runner import (
    benchmark_alignment,
    benchmark_deconvolution,
    benchmark_domain,
    to_dataframe,
)

__all__ = [
    "Method",
    "DeconvolutionMethod",
    "AlignmentMethod",
    "DomainMethod",
    "IsolatedEnvMethod",
    "MethodUnavailable",
    "DECONVOLUTION_METHODS",
    "ALIGNMENT_METHODS",
    "benchmark_deconvolution",
    "benchmark_alignment",
    "benchmark_domain",
    "to_dataframe",
]
