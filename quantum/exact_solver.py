"""
Brute-force exact QUBO solver for Q-Signal.
Validates QUBO coefficients and provides ground-truth solutions for small instances (n ≤ 20).
"""

from __future__ import annotations

import time
from itertools import product
from typing import List, Optional, Tuple

import numpy as np


def brute_force_qubo(
    Q: np.ndarray,
    max_bits: int = 20,
) -> Tuple[List[int], float]:
    """
    Enumerate all 2^n bitstrings and return the one with minimum energy.

    Args:
        Q: (n × n) QUBO matrix (upper-triangular form).
        max_bits: Maximum n allowed; raises ValueError if exceeded.

    Returns:
        best_bits: List[int] of length n.
        best_energy: Minimum QUBO energy found.
    """
    n = Q.shape[0]
    if n > max_bits:
        raise ValueError(
            f"Brute-force solver limited to n≤{max_bits} variables, got n={n}. "
            "Use QAOA or another heuristic for larger instances."
        )

    best_bits: List[int] = [0] * n
    best_energy = float("inf")

    for bits in product([0, 1], repeat=n):
        x = np.array(bits, dtype=float)
        energy = float(x @ Q @ x)
        if energy < best_energy:
            best_energy = energy
            best_bits = list(bits)

    return best_bits, best_energy


def enumerate_all(
    Q: np.ndarray,
    max_bits: int = 20,
) -> List[Tuple[List[int], float]]:
    """
    Return all (bits, energy) pairs sorted by energy ascending.
    Useful for computing approximation ratios.
    """
    n = Q.shape[0]
    if n > max_bits:
        raise ValueError(f"Too many variables: {n} > {max_bits}")

    results = []
    for bits in product([0, 1], repeat=n):
        x = np.array(bits, dtype=float)
        energy = float(x @ Q @ x)
        results.append((list(bits), energy))

    results.sort(key=lambda r: r[1])
    return results


def brute_force_timed(
    Q: np.ndarray,
    max_bits: int = 20,
) -> dict:
    """
    Run brute-force solver and return rich result dict.
    """
    t0 = time.perf_counter()
    bits, energy = brute_force_qubo(Q, max_bits=max_bits)
    elapsed = time.perf_counter() - t0
    return {
        "bits": bits,
        "energy": energy,
        "elapsed_s": elapsed,
        "n_evaluated": 2 ** Q.shape[0],
        "backend": "brute_force",
        "feasible": True,
    }
