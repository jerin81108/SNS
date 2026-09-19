"""
Unit tests for the QUBO builder and exact solver.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np

from qsignal.simulator import TrafficSimulator
from qsignal.models import PHASE_NS, PHASE_EW
from quantum.qubo import build_qubo, qubo_energy
from quantum.exact_solver import brute_force_qubo, enumerate_all


@pytest.fixture
def simple_state():
    """A 2-intersection simulator with known queue state."""
    sim = TrafficSimulator(n_intersections=4, profile="normal", seed=42)
    # Manually set queues so we can predict QUBO structure
    for approach in sim.state.approaches.values():
        if approach.movement == "NS":
            approach.queue = 5.0
        else:
            approach.queue = 2.0
    return sim.state


# ── QUBO shape ─────────────────────────────────────────────────────────────────
def test_qubo_shape(simple_state):
    """QUBO matrix should be n×n where n = number of intersections."""
    Q, offset = build_qubo(simple_state)
    n = len(simple_state.intersections)
    assert Q.shape == (n, n), f"Expected ({n},{n}), got {Q.shape}"


# ── Exact solver returns valid bits ───────────────────────────────────────────
def test_brute_force_returns_valid_bits(simple_state):
    Q, _ = build_qubo(simple_state)
    n = Q.shape[0]
    bits, energy = brute_force_qubo(Q)
    assert len(bits) == n
    assert all(b in (0, 1) for b in bits), "All bits must be 0 or 1"


# ── Best energy is actually the minimum ───────────────────────────────────────
def test_brute_force_finds_minimum(simple_state):
    Q, _ = build_qubo(simple_state)
    bits, best_energy = brute_force_qubo(Q)

    # Enumerate all and verify
    all_results = enumerate_all(Q)
    assert abs(all_results[0][1] - best_energy) < 1e-9, \
        f"brute_force_qubo energy {best_energy} != enumerated minimum {all_results[0][1]}"


# ── QUBO energy function consistency ──────────────────────────────────────────
def test_qubo_energy_consistent(simple_state):
    Q, _ = build_qubo(simple_state)
    bits, expected_energy = brute_force_qubo(Q)
    computed = qubo_energy(Q, bits)
    assert abs(computed - expected_energy) < 1e-9, \
        f"qubo_energy mismatch: {computed} vs {expected_energy}"


# ── NS-heavy demand should prefer NS phase ─────────────────────────────────────
def test_ns_heavy_prefers_ns():
    """When NS queues are much larger than EW, optimal assignment should prefer NS (x_i=0)."""
    sim = TrafficSimulator(n_intersections=4, profile="normal", seed=42)
    for approach in sim.state.approaches.values():
        approach.queue = 20.0 if approach.movement == "NS" else 0.5
    Q, _ = build_qubo(sim.state)
    bits, _ = brute_force_qubo(Q)
    # Most bits should be 0 (NS phase)
    ns_count = sum(1 for b in bits if b == 0)
    assert ns_count >= len(bits) // 2, \
        f"Expected NS majority with heavy NS demand, got bits={bits}"


# ── Small instance manual calculation ─────────────────────────────────────────
def test_single_intersection_manual():
    """
    For n=1 with Q = [[c]], x=0 → E=0, x=1 → E=c.
    If c > 0, optimal should be x=0. If c < 0, optimal should be x=1.
    """
    Q_pos = np.array([[3.0]])
    bits, energy = brute_force_qubo(Q_pos)
    assert bits == [0] and abs(energy) < 1e-9, f"Expected [0] with energy~0, got {bits}, {energy}"

    Q_neg = np.array([[-3.0]])
    bits, energy = brute_force_qubo(Q_neg)
    assert bits == [1] and abs(energy - (-3.0)) < 1e-9, f"Expected [1] with energy~-3, got {bits}, {energy}"


# ── Exact solver too large ─────────────────────────────────────────────────────
def test_brute_force_too_large():
    """Should raise ValueError for n > 20."""
    Q = np.zeros((21, 21))
    with pytest.raises(ValueError, match="Brute-force solver"):
        brute_force_qubo(Q)
