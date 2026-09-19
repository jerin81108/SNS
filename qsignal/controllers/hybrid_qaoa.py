"""
Hybrid QAOA controller for Q-Signal.
Unified solve_action() interface for all three controller modes.
QAOA result is always passed through the safety validator before deployment.
"""

from __future__ import annotations

import time
from typing import Dict, Optional

from qsignal.models import TrafficState, SignalPlan, PHASE_NS, PHASE_EW
from qsignal.controllers.fixed_time import FixedTimeController
from qsignal.controllers.pressure import PressureController
from qsignal.safety import SafetyValidator


def solve_action(
    state: TrafficState,
    mode: str = "qaoa",
    config: Optional[dict] = None,
    graph=None,
) -> SignalPlan:
    """
    Unified controller dispatch.

    mode: "fixed" | "pressure" | "qaoa"
    Falls back to pressure if QAOA fails or times out.
    """
    if mode == "fixed":
        ctrl = FixedTimeController(green_duration=config.get("green_duration", 20) if config else 20)
        return ctrl.compute_action(state)

    if mode == "pressure":
        ctrl = PressureController()
        return ctrl.compute_action(state)

    if mode == "qaoa":
        return _qaoa_action(state, config=config or {}, graph=graph)

    raise ValueError(f"Unsupported controller mode: {mode!r}")


# ---------------------------------------------------------------------------
# QAOA dispatch
# ---------------------------------------------------------------------------

def _qaoa_action(
    state: TrafficState,
    config: dict,
    graph=None,
) -> SignalPlan:
    """
    Build QUBO → solve with QAOA (or exact fallback) → validate.
    """
    from quantum.qubo import build_qubo
    from quantum.exact_solver import brute_force_qubo

    shots = config.get("shots", 512)
    depth = config.get("depth", 2)
    time_budget = config.get("time_budget_s", 5.0)
    fallback = config.get("fallback", "exact")

    node_ids = list(state.intersections.keys())
    Q, offset = build_qubo(state, graph=graph, config=config)

    # Exact brute-force reference (always computed for small networks)
    exact_bits, exact_energy = brute_force_qubo(Q)

    # Attempt QAOA
    qaoa_bits = None
    qaoa_energy = None
    solver_used = "exact"
    t0 = time.time()

    try:
        from quantum.qiskit_solver import solve_qaoa
        result = solve_qaoa(Q, shots=shots, depth=depth, time_budget=time_budget)
        if result["feasible"] and (time.time() - t0) <= time_budget:
            qaoa_bits = result["bits"]
            qaoa_energy = result["energy"]
            solver_used = "qaoa"
    except Exception:
        pass  # Fall through to exact fallback

    # Use exact if QAOA failed or timed out
    bits = qaoa_bits if qaoa_bits is not None else list(exact_bits)
    energy = qaoa_energy if qaoa_energy is not None else exact_energy

    # Compute QAOA gap
    gap_pct: Optional[float] = None
    if qaoa_energy is not None and exact_energy is not None and abs(exact_energy) > 1e-9:
        gap_pct = 100.0 * (qaoa_energy - exact_energy) / abs(exact_energy)

    # Convert bits to assignments
    assignments: Dict[str, int] = {}
    for i, node_id in enumerate(node_ids):
        assignments[node_id] = PHASE_EW if (i < len(bits) and bits[i] == 1) else PHASE_NS

    plan = SignalPlan(
        assignments=assignments,
        objective_value=float(energy) if energy is not None else 0.0,
        feasible=True,
        solver=solver_used,
        qaoa_energy=energy,
        qaoa_gap_pct=gap_pct,
        bits=bits,
    )

    # Safety validation
    validator = SafetyValidator()
    safe_plan, _ = validator.validate(plan, state)
    return safe_plan
