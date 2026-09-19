"""
Unit and integration tests for the Q-Signal traffic simulator.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from qsignal.simulator import TrafficSimulator
from qsignal.models import PHASE_NS, PHASE_EW


@pytest.fixture
def sim():
    return TrafficSimulator(n_intersections=4, profile="normal", seed=0)


# ── Zero-arrival test ──────────────────────────────────────────────────────────
def test_zero_arrivals_zero_queues():
    """With demand rate = 0, all queues must stay at 0."""
    s = TrafficSimulator(n_intersections=4, profile="low", seed=0)
    # Override demand rates to zero
    s.demand.base_rates = {"NS": 0.0, "EW": 0.0}
    s.step(60)
    for approach in s.state.approaches.values():
        assert approach.queue == pytest.approx(0.0, abs=1e-6), \
            f"Approach {approach.approach_id} queue is not zero"


# ── Service capacity test ───────────────────────────────────────────────────────
def test_service_capacity_limited():
    """
    With 10 vehicles queued on a green approach and 0.5 veh/s service rate,
    exactly 0.5 vehicles are served per second (fractional, accumulated).
    """
    s = TrafficSimulator(n_intersections=4, profile="normal", seed=0)
    # Manually inject vehicles into one approach
    nid = s.node_ids[0]
    approach_id = f"{nid}_NS"
    approach = s.state.approaches[approach_id]
    approach.queue = 10.0

    # Force green on that intersection for NS
    s.state.intersections[nid].phase = PHASE_NS
    s.state.intersections[nid].elapsed_green = 20  # past min green

    # Turn off all demand
    s.demand.base_rates = {"NS": 0.0, "EW": 0.0}

    # Step 2 seconds
    s.step(2)

    # 2 seconds × 0.5 veh/s = 1 vehicle served
    assert approach.queue == pytest.approx(9.0, abs=0.5), \
        f"Expected ~9.0 vehicles in queue after 2s service, got {approach.queue}"


# ── Queue non-negativity ────────────────────────────────────────────────────────
def test_queues_never_negative(sim):
    """Property test: queues must never go below 0."""
    for _ in range(300):
        sim.step(1)
        for approach in sim.state.approaches.values():
            assert approach.queue >= 0.0, \
                f"Negative queue on {approach.approach_id}: {approach.queue}"


# ── Signal phase validity ───────────────────────────────────────────────────────
def test_phases_always_valid(sim):
    """All intersections should always have a valid phase (0, 1, 2, or 3)."""
    valid = {PHASE_NS, PHASE_EW, 2, 3}
    for _ in range(100):
        sim.step(1)
        for nid, inter in sim.state.intersections.items():
            assert inter.phase in valid, f"{nid} has invalid phase {inter.phase}"


# ── Min-green enforcement ───────────────────────────────────────────────────────
def test_min_green_respected(sim):
    """
    When a plan requests a switch, min_green must be respected.
    """
    from qsignal.safety import SafetyValidator
    from qsignal.models import SignalPlan

    validator = SafetyValidator(min_green=10)
    nid = sim.node_ids[0]
    inter = sim.state.intersections[nid]
    inter.phase = PHASE_NS
    inter.elapsed_green = 3  # below min_green

    # Propose switching to EW
    plan = SignalPlan(assignments={nid: PHASE_EW}, solver="test")
    safe_plan, all_safe = validator.validate(plan, sim.state)

    assert not all_safe, "Should have flagged a min_green violation"
    assert safe_plan.assignments[nid] == PHASE_NS, "Should have kept NS phase"


# ── Integration: simulation advances ───────────────────────────────────────────
def test_simulation_advances_time(sim):
    """Calling step() should advance simulation time."""
    assert sim.state.time_s == 0
    sim.step(10)
    assert sim.state.time_s == 10


# ── Integration: metrics produced ──────────────────────────────────────────────
def test_metrics_produced(sim):
    """After stepping, record_metrics should return a valid snapshot."""
    sim.step(30)
    snap = sim.record_metrics("test_ctrl")
    assert snap.time_s == 30
    assert snap.controller == "test_ctrl"
    assert snap.mean_queue >= 0
    assert snap.max_queue >= snap.mean_queue


# ── Reset test ─────────────────────────────────────────────────────────────────
def test_reset_clears_state(sim):
    """After reset, time should be 0 and queues should be 0."""
    sim.step(50)
    sim.reset()
    assert sim.state.time_s == 0
    for approach in sim.state.approaches.values():
        assert approach.queue == 0.0
