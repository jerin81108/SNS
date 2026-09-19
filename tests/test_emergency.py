"""
Emergency corridor tests for Q-Signal.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from qsignal.simulator import TrafficSimulator
from qsignal.emergency import EmergencyCorridorManager
from qsignal.models import PHASE_NS, PHASE_EW, SignalPlan
from qsignal.safety import SafetyValidator


@pytest.fixture
def sim():
    return TrafficSimulator(n_intersections=4, seed=0)


@pytest.fixture
def em_mgr(sim):
    return EmergencyCorridorManager(sim, clearance_buffer_s=10, advance_reservations=2)


# ── Route generation ───────────────────────────────────────────────────────────
def test_route_generated(sim, em_mgr):
    """Activating emergency should produce a valid route."""
    node_ids = sim.node_ids
    origin = node_ids[0]
    dest = node_ids[-1]
    em = em_mgr.activate(origin, dest)
    assert em is not None, "Should find a route"
    assert em.route[0] == origin
    assert em.route[-1] == dest


def test_route_connected(sim, em_mgr):
    """Every consecutive pair in the route must be connected in the graph."""
    node_ids = sim.node_ids
    em = em_mgr.activate(node_ids[0], node_ids[-1])
    if em is None:
        pytest.skip("No route available for this network")
    for i in range(len(em.route) - 1):
        u, v = em.route[i], em.route[i + 1]
        assert sim.graph.has_edge(u, v), f"Edge {u}→{v} missing in graph"


# ── Phase reservation ──────────────────────────────────────────────────────────
def test_phases_reserved(sim, em_mgr):
    """After activation, intersections along the route should be reserved."""
    node_ids = sim.node_ids
    em = em_mgr.activate(node_ids[0], node_ids[-1])
    if em is None:
        pytest.skip("No route")
    # At least the first intersection should be reserved
    first = em.route[0]
    inter = sim.state.intersections[first]
    assert inter.is_reserved, f"First route node {first} should be reserved"


# ── Emergency beats optimizer ─────────────────────────────────────────────────
def test_emergency_beats_optimizer(sim, em_mgr):
    """Safety validator must preserve emergency reservation against any optimizer proposal."""
    node_ids = sim.node_ids
    em = em_mgr.activate(node_ids[0], node_ids[-1])
    if em is None:
        pytest.skip("No route")

    validator = SafetyValidator(min_green=10)
    reserved_node = em.route[0]
    inter = sim.state.intersections[reserved_node]
    required_phase = inter.reserved_phase
    other_phase = PHASE_EW if required_phase == PHASE_NS else PHASE_NS

    # Force enough elapsed green so min-green isn't blocking
    inter.elapsed_green = 20

    plan = SignalPlan(assignments={reserved_node: other_phase}, solver="qaoa")
    safe_plan, all_safe = validator.validate(plan, sim.state)

    assert not all_safe, "Should have rejected the phase change due to reservation"
    assert safe_plan.assignments[reserved_node] == required_phase


# ── Invalid route (same origin/destination) ────────────────────────────────────
def test_same_origin_dest_returns_none(sim, em_mgr):
    nid = sim.node_ids[0]
    em = em_mgr.activate(nid, nid)
    assert em is None


# ── Invalid destination not in network ────────────────────────────────────────
def test_invalid_destination(sim, em_mgr):
    em = em_mgr.activate(sim.node_ids[0], "NONEXISTENT")
    assert em is None


# ── Corridor release ───────────────────────────────────────────────────────────
def test_corridor_release(sim, em_mgr):
    """After releasing the corridor, no intersection should be reserved."""
    node_ids = sim.node_ids
    em = em_mgr.activate(node_ids[0], node_ids[-1])
    if em is None:
        pytest.skip("No route")
    em_mgr.release()
    for nid in em.route:
        inter = sim.state.intersections.get(nid)
        if inter:
            assert not inter.is_reserved, f"{nid} should not be reserved after release"
