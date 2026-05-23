"""Lightweight wall-clock timing for runtime/scalability benchmarks (PLAN.md 2.3)."""

from __future__ import annotations

import time
from contextlib import contextmanager

__all__ = ["Timer", "timed"]


class Timer:
    """Context manager that records elapsed wall-clock seconds.

    >>> with Timer() as t:
    ...     pass
    >>> t.elapsed >= 0
    True
    """

    def __init__(self) -> None:
        self.start: float | None = None
        self.elapsed: float = 0.0

    def __enter__(self) -> Timer:
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed = time.perf_counter() - self.start


@contextmanager
def timed(label: str):
    """Time a block and print ``label: <seconds>s`` on exit; yields the live ``Timer``."""
    timer = Timer()
    with timer:
        yield timer
    print(f"{label}: {timer.elapsed:.4f}s")
