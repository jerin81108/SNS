"""
Qiskit Aer QAOA solver for Q-Signal.
Isolated from all other modules to minimize import-failure blast radius.
Qiskit 1.x / 2.x API used (pinned in requirements.txt).
"""

from __future__ import annotations

import time
import warnings
from typing import List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Lazy imports – wrapped in try/except so the rest of the app still works
# if Qiskit Aer is not available.
# ---------------------------------------------------------------------------
try:
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from qiskit_aer import AerSimulator
    _QISKIT_AVAILABLE = True
except ImportError:
    _QISKIT_AVAILABLE = False


def is_available() -> bool:
    return _QISKIT_AVAILABLE


def solve_qaoa(
    Q: np.ndarray,
    shots: int = 512,
    depth: int = 2,
    time_budget: float = 5.0,
) -> dict:
    """
    Solve a QUBO with QAOA on a local Aer simulator.

    Args:
        Q: (n × n) QUBO matrix.
        shots: Number of measurement shots.
        depth: QAOA circuit depth (p layers).
        time_budget: Maximum seconds allowed before falling back.

    Returns dict with keys:
        bits, energy, feasible, backend, shots, elapsed_s
    """
    if not _QISKIT_AVAILABLE:
        return _fallback_result(Q, "qiskit_not_installed")

    n = Q.shape[0]
    if n == 0:
        return {"bits": [], "energy": 0.0, "feasible": True, "backend": "aer", "shots": shots}

    t0 = time.perf_counter()

    try:
        # Classical COBYLA optimizer for QAOA angles
        from scipy.optimize import minimize as scipy_minimize

        # Pauli operator from QUBO
        # Convert Q to Ising (Z-basis): x_i = (1 - z_i) / 2
        # Energy = x^T Q x → Ising Hamiltonian
        h, J, offset = _qubo_to_ising(Q)

        # Build parameterized QAOA circuit
        gamma_params = ParameterVector("γ", depth)
        beta_params = ParameterVector("β", depth)
        qc = _build_qaoa_circuit(n, h, J, gamma_params, beta_params, depth)

        # Objective: expectation of QUBO energy from sampled bitstrings
        simulator = AerSimulator()

        def objective(params: np.ndarray) -> float:
            gamma_vals = params[:depth]
            beta_vals = params[depth:]
            param_dict = {}
            for k in range(depth):
                param_dict[gamma_params[k]] = float(gamma_vals[k])
                param_dict[beta_params[k]] = float(beta_vals[k])
            bound = qc.assign_parameters(param_dict)
            bound.measure_all()
            job = simulator.run(bound, shots=shots)
            counts = job.result().get_counts()
            return _expectation_from_counts(counts, Q, n)

        # Check time budget before optimization
        if time.perf_counter() - t0 > time_budget:
            return _fallback_result(Q, "time_budget_exceeded")

        # Initial parameters
        rng = np.random.default_rng(42)
        x0 = rng.uniform(0, np.pi, size=2 * depth)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = scipy_minimize(
                objective,
                x0,
                method="COBYLA",
                options={"maxiter": 100, "rhobeg": 0.5},
            )

        if time.perf_counter() - t0 > time_budget:
            return _fallback_result(Q, "time_budget_exceeded")

        # Sample final circuit with optimized parameters
        opt_params = result.x
        param_dict = {}
        for k in range(depth):
            param_dict[gamma_params[k]] = float(opt_params[k])
            param_dict[beta_params[k]] = float(opt_params[depth + k])
        final_qc = qc.assign_parameters(param_dict)
        final_qc.measure_all()

        job = simulator.run(final_qc, shots=shots)
        counts = job.result().get_counts()

        # Pick the most frequent feasible bitstring
        best_bits, best_energy = _best_from_counts(counts, Q, n)
        elapsed = time.perf_counter() - t0

        return {
            "bits": best_bits,
            "energy": best_energy,
            "feasible": True,
            "backend": "aer_simulator",
            "shots": shots,
            "elapsed_s": elapsed,
            "counts": dict(sorted(counts.items(), key=lambda x: -x[1])[:10]),
        }

    except Exception as exc:
        return _fallback_result(Q, f"exception: {exc}")


# ---------------------------------------------------------------------------
# QAOA circuit builder
# ---------------------------------------------------------------------------

def _build_qaoa_circuit(
    n: int,
    h: np.ndarray,
    J: np.ndarray,
    gamma_params: "ParameterVector",
    beta_params: "ParameterVector",
    depth: int,
) -> "QuantumCircuit":
    """Build a parameterized QAOA circuit for the given Ising problem."""
    qc = QuantumCircuit(n)

    # Initial state: uniform superposition
    qc.h(range(n))

    for p in range(depth):
        # Cost layer (phase separation)
        gamma = gamma_params[p]

        # Single-qubit Z terms
        for i in range(n):
            if abs(h[i]) > 1e-10:
                qc.rz(2.0 * gamma * h[i], i)

        # Two-qubit ZZ terms
        for i in range(n):
            for j in range(i + 1, n):
                if abs(J[i, j]) > 1e-10:
                    qc.cx(i, j)
                    qc.rz(2.0 * gamma * J[i, j], j)
                    qc.cx(i, j)

        # Mixer layer (X rotations)
        beta = beta_params[p]
        qc.rx(2.0 * beta, range(n))

    return qc


# ---------------------------------------------------------------------------
# Ising conversion
# ---------------------------------------------------------------------------

def _qubo_to_ising(
    Q: np.ndarray,
) -> tuple:
    """
    Convert QUBO Q to Ising (h, J, offset).
    x_i = (1 - z_i) / 2, z_i ∈ {-1, +1}
    """
    n = Q.shape[0]
    # Make Q symmetric
    Qs = (Q + Q.T) / 2.0
    np.fill_diagonal(Qs, np.diag(Q))

    h = np.zeros(n)
    J = np.zeros((n, n))
    offset = 0.0

    for i in range(n):
        for j in range(i, n):
            if i == j:
                # x_i = (1-z_i)/2 → Q[i,i]*x_i = Q[i,i]*(1-z_i)/2
                h[i] -= Qs[i, i] / 2.0
                offset += Qs[i, i] / 2.0
            else:
                # Q[i,j]*x_i*x_j = Q[i,j]*(1-z_i)*(1-z_j)/4
                J[i, j] += Qs[i, j] / 4.0
                h[i] -= Qs[i, j] / 4.0
                h[j] -= Qs[i, j] / 4.0
                offset += Qs[i, j] / 4.0

    return h, J, offset


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _expectation_from_counts(counts: dict, Q: np.ndarray, n: int) -> float:
    """Compute expectation value of QUBO energy from measurement counts."""
    total_shots = sum(counts.values())
    if total_shots == 0:
        return 0.0
    exp = 0.0
    for bitstr, cnt in counts.items():
        # Qiskit bitstrings are reversed (qubit 0 is rightmost)
        bits = [int(b) for b in reversed(bitstr.replace(" ", ""))]
        bits = bits[:n] + [0] * max(0, n - len(bits))
        x = np.array(bits[:n], dtype=float)
        energy = float(x @ Q @ x)
        exp += energy * cnt / total_shots
    return exp


def _best_from_counts(counts: dict, Q: np.ndarray, n: int):
    """Return the bits and energy of the lowest-energy measured bitstring."""
    best_bits = [0] * n
    best_energy = float("inf")
    for bitstr in counts:
        bits = [int(b) for b in reversed(bitstr.replace(" ", ""))]
        bits = bits[:n] + [0] * max(0, n - len(bits))
        bits = bits[:n]
        x = np.array(bits, dtype=float)
        energy = float(x @ Q @ x)
        if energy < best_energy:
            best_energy = energy
            best_bits = bits
    return best_bits, best_energy


def _fallback_result(Q: np.ndarray, reason: str) -> dict:
    """Return a fallback result using greedy assignment."""
    n = Q.shape[0]
    # Greedy: for each variable pick the value that minimises diagonal energy
    bits = [1 if Q[i, i] < 0 else 0 for i in range(n)]
    x = np.array(bits, dtype=float)
    energy = float(x @ Q @ x)
    return {
        "bits": bits,
        "energy": energy,
        "feasible": False,
        "backend": "fallback",
        "shots": 0,
        "reason": reason,
        "elapsed_s": 0.0,
    }
