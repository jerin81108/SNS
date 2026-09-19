"""
Safety validator tests for Q-Signal.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from qsignal.simulator import TrafficSimulator
from qsignal.models import SignalPlan, PHASE_NS, PHASE_EW
from qsignal.safety import SafetyValidator


@pytest.fixture
def sim():
    return TrafficSimulator(n_intersections=4, seed=0)


@pytest.fixture
def validator():
    return SafetyValidator(min_green=10, yellow_s=3, all_red_s=2)


# ── Valid plan passes ──────────────────────────────────────────────────────────
def test_valid_plan_passes(sim, validator):
    """A plan with correct phase and sufficient elapsed green should pass unchanged."""
    nid = sim.node_ids[0]
    inter = sim.state.intersections[nid]
    inter.phase = PHASE_NS
    inter.elapsed_green = 15  # > min_green

    plan = SignalPlan(assignments={nid: PHASE_EW}, solver="test")
    safe_plan, all_safe = validator.validate(plan, sim.state)
    assert all_safe
    assert safe_plan.assignments[nid] == PHASE_EW


# ── Min-green violation caught ────────────────────────────────────────────────
def test_min_green_violation(sim, validator):
    nid = sim.node_ids[0]
    inter = sim.state.intersections[nid]
    inter.phase = PHASE_NS
    inter.elapsed_green = 3  # below min_green=10

    plan = SignalPlan(assignments={nid: PHASE_EW}, solver="test")
    safe_plan, all_safe = validator.validate(plan, sim.state)

    assert not all_safe
    assert safe_plan.assignments[nid] == PHASE_NS
    violations = validator.get_violations()
    assert any("min_green" in v["reason"] for v in violations)


# ── Clearance interval blocks switch ─────────────────────────────────────────
def test_clearance_blocks_switch(sim, validator):
    nid = sim.node_ids[0]
    inter = sim.state.intersections[nid]
    inter.yellow_remaining = 2  # in clearance

    plan = SignalPlan(assignments={nid: PHASE_EW}, solver="test")
    safe_plan, all_safe = validator.validate(plan, sim.state)

    assert not all_safe
    assert safe_plan.assignments[nid] == inter.phase
    assert any("clearance" in v["reason"] for v in validator.get_violations())


# ── Emergency reservation cannot be overridden ────────────────────────────────
def test_emergency_reservation_locks_phase(sim, validator):
    nid = sim.node_ids[0]
    inter = sim.state.intersections[nid]
    inter.phase = PHASE_NS
    inter.elapsed_green = 20
    inter.reserved_phase = PHASE_NS
    inter.reserved_until = 9999

    # Try to override with EW
    plan = SignalPlan(assignments={nid: PHASE_EW}, solver="test")
    safe_plan, all_safe = validator.validate(plan, sim.state)

    assert not all_safe
    assert safe_plan.assignments[nid] == PHASE_NS
    violations = validator.get_violations()
    assert any("emergency" in v["reason"] for v in violations)


# ── Conflict-free always true in two-phase model ──────────────────────────────
def test_no_conflicts(sim, validator):
    """Conflict-free check should always return True in the binary model."""
    sim.step(30)
    assert validator.check_conflict_free(sim.state)


# ── Invalid phase value caught ───────────────────────────────────────────────
def test_invalid_phase_rejected(sim, validator):
    nid = sim.node_ids[0]
    inter = sim.state.intersections[nid]
    inter.elapsed_green = 20

    plan = SignalPlan(assignments={nid: 99}, solver="test")  # invalid
    safe_plan, all_safe = validator.validate(plan, sim.state)
    assert not all_safe
    assert safe_plan.assignments[nid] in (PHASE_NS, PHASE_EW)
