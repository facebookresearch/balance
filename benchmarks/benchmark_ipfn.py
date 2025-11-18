# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import copy
import time
from typing import Callable, Sequence

import numpy as np

from balance.weighting_methods.rake import _run_ipf_numpy

try:
    from ipfn import ipfn as ipfn_module
except ImportError:  # pragma: no cover - optional dependency for benchmarking
    ipfn_module = None


def _build_problem(seed: int = 0) -> tuple[np.ndarray, list[np.ndarray]]:
    """Construct a moderately sized contingency table and consistent margins."""

    rng = np.random.default_rng(seed)
    shape = (8, 10, 12)
    table = rng.uniform(0.1, 5.0, size=shape)

    margins: list[np.ndarray] = []
    for axis in range(table.ndim):
        margin = table.sum(axis=tuple(i for i in range(table.ndim) if i != axis))
        # Introduce mild perturbations while keeping totals consistent.
        noise = rng.uniform(0.95, 1.05, size=margin.shape)
        margin = margin * noise
        margin *= table.sum() / margin.sum()
        margins.append(margin)
    return table, margins


def _timeit(func: Callable[[], np.ndarray], repeat: int = 7) -> float:
    start = time.perf_counter()
    for _ in range(repeat):
        func()
    end = time.perf_counter()
    return (end - start) / repeat


def _run_ipfn_lib(original: np.ndarray, margins: Sequence[np.ndarray]) -> np.ndarray:
    if ipfn_module is None:
        raise RuntimeError(
            "The `ipfn` package is not installed. Install it with `pip install ipfn`"
            " to run the lib benchmark."
        )

    dims = [[axis] for axis in range(original.ndim)]
    solver = ipfn_module.ipfn(
        copy.deepcopy(original),
        [np.array(m, copy=True) for m in margins],
        dims,
        convergence_rate=5e-7,
        max_iteration=1000,
        rate_tolerance=0.0,
        verbose=0,
    )
    return solver.iteration()


def _run_ipfn_numpy(original: np.ndarray, margins: Sequence[np.ndarray]) -> np.ndarray:
    solution, _, _ = _run_ipf_numpy(
        np.array(original, copy=True),
        [np.array(m, copy=True) for m in margins],
        convergence_rate=5e-7,
        max_iteration=1000,
        rate_tolerance=0.0,
    )
    return solution


def main() -> None:
    original, margins = _build_problem()

    numpy_solution = _run_ipfn_numpy(original, margins)
    if ipfn_module is not None:
        lib_solution = _run_ipfn_lib(original, margins)
        np.testing.assert_allclose(lib_solution, numpy_solution, atol=1e-6)

    numpy_timing = _timeit(lambda: _run_ipfn_numpy(original, margins))

    if ipfn_module is None:
        print("ipfn package not installed; only NumPy implementation timing available.")
        print(f"NumPy IPF solver: {numpy_timing * 1000:.2f} ms per run")
        return

    lib_timing = _timeit(lambda: _run_ipfn_lib(original, margins))

    print("Iterative proportional fitting benchmark (5-run average)")
    print(f"lib ipfn.ipfn solver: {lib_timing * 1000:.2f} ms per run")
    print(f"NumPy _run_ipf_numpy solver: {numpy_timing * 1000:.2f} ms per run")
    print(f"Speed-up: {lib_timing / numpy_timing:.2f}x faster")


if __name__ == "__main__":  # pragma: no cover - convenience entry point
    main()
