"""
PennyLane QAOA solver for Q-Signal (optional extension).
Implements the same interface as qiskit_solver.py.
"""

from __future__ import annotations

import time
from typing import List

import numpy as np

try:
    import pennylane as qml
    from pennylane import numpy as pnp
    _PENNYLANE_AVAILABLE = True
except ImportError:
    _PENNYLANE_AVAILABLE = False


def is_available() -> bool:
    return _PENNYLANE_AVAILABLE


def solve_qaoa(
    Q: np.ndarray,
    shots: int = 512,
    depth: int = 2,
    time_budget: float = 5.0,
) -> dict:
    """
    Solve QUBO with PennyLane QAOA (optional backend).
    Falls back gracefully if PennyLane is not installed.
    """
    if not _PENNYLANE_AVAILABLE:
        return {
            "bits": [0] * Q.shape[0],
            "energy": 0.0,
            "feasible": False,
            "backend": "pennylane_not_installed",
            "shots": 0,
            "elapsed_s": 0.0,
        }

    t0 = time.perf_counter()
    n = Q.shape[0]

    try:
        dev = qml.device("default.qubit", wires=n, shots=shots)

        # Convert QUBO to cost Hamiltonian
        coeffs, obs = [], []
        for i in range(n):
            for j in range(i, n):
                if i == j and abs(Q[i, i]) > 1e-10:
                    coeffs.append(Q[i, i])
                    obs.append(qml.PauliZ(i))
                elif i != j and abs(Q[i, j]) > 1e-10:
                    coeffs.append(Q[i, j])
                    obs.append(qml.PauliZ(i) @ qml.PauliZ(j))

        H_cost = qml.Hamiltonian(coeffs, obs) if coeffs else qml.Hamiltonian([0.0], [qml.Identity(0)])

        @qml.qnode(dev)
        def circuit(gamma, beta):
            # Initial superposition
            for i in range(n):
                qml.Hadamard(wires=i)
            # QAOA layers
            for layer in range(depth):
                qml.qaoa.cost_layer(gamma[layer], H_cost)
                qml.qaoa.mixer_layer(beta[layer], qml.qaoa.x_mixer(range(n)))
            return qml.sample(wires=range(n))

        # Optimize
        gamma = pnp.array([0.5] * depth, requires_grad=True)
        beta = pnp.array([0.3] * depth, requires_grad=True)
        opt = qml.AdamOptimizer(stepsize=0.1)

        def cost_fn(gamma, beta):
            samples = circuit(gamma, beta)
            energies = [float(np.array(s) @ Q @ np.array(s)) for s in samples]
            return np.mean(energies)

        for _ in range(30):
            gamma, beta = opt.step(cost_fn, gamma, beta)
            if time.perf_counter() - t0 > time_budget:
                break

        # Final sample
        samples = circuit(gamma, beta)
        best_bits, best_energy = None, float("inf")
        for s in samples:
            bits = list(map(int, s))
            x = np.array(bits, dtype=float)
            e = float(x @ Q @ x)
            if e < best_energy:
                best_energy = e
                best_bits = bits

        return {
            "bits": best_bits or [0] * n,
            "energy": best_energy,
            "feasible": True,
            "backend": "pennylane",
            "shots": shots,
            "elapsed_s": time.perf_counter() - t0,
        }

    except Exception as exc:
        return {
            "bits": [0] * n,
            "energy": 0.0,
            "feasible": False,
            "backend": "pennylane_error",
            "shots": 0,
            "elapsed_s": time.perf_counter() - t0,
            "reason": str(exc),
        }
