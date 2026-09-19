"""
Discrete-time traffic simulator for Q-Signal.
Operates at 1-second ticks. No Streamlit calls anywhere in this module.
"""

from __future__ import annotations

import copy
import random
from typing import Callable, Dict, List, Optional, Tuple

import networkx as nx

from qsignal.models import (
    Intersection,
    Approach,
    Vehicle,
    TrafficState,
    EmergencyRequest,
    MetricsSnapshot,
    PHASE_NS,
    PHASE_EW,
    PHASE_YELLOW,
    PHASE_ALL_RED,
    PHASE_MOVEMENT,
)
from qsignal.demand import DemandGenerator
from qsignal.network import build_grid_network, find_shortest_path, get_edge_direction


class TrafficSimulator:
    """
    Custom discrete-time traffic simulator.

    Simulation loop per tick:
      1. Generate arrivals (demand generator)
      2. Apply incidents (lane closures, spikes)
      3. Service green approaches
      4. Advance vehicles through network
      5. Record metrics
    """

    def __init__(
        self,
        n_intersections: int = 4,
        profile: str = "normal",
        seed: int = 42,
        config: Optional[dict] = None,
    ):
        cfg = config or {}
        self.min_green: int = cfg.get("min_green_s", 10)
        self.max_green: int = cfg.get("max_green_s", 40)
        self.yellow_s: int = cfg.get("yellow_s", 3)
        self.all_red_s: int = cfg.get("all_red_s", 2)
        self.saturation_flow: float = cfg.get("saturation_flow", 0.5)
        self.idle_fuel: float = cfg.get("idle_fuel_L_per_s", 0.00083)
        self.co2_per_litre: float = cfg.get("co2_per_litre", 2.31)

        self.n_intersections = n_intersections
        self.seed = seed
        self._rng = random.Random(seed)

        # Build network
        self.graph, intersections, approaches = build_grid_network(
            n=n_intersections,
            min_green=self.min_green,
            max_green=self.max_green,
            saturation_flow=self.saturation_flow,
        )
        self.node_ids: List[str] = list(intersections.keys())

        # Demand generator
        self.demand = DemandGenerator(profile=profile, seed=seed)

        # Live state
        self.state = TrafficState(
            time_s=0,
            intersections=intersections,
            approaches=approaches,
            arrival_buffers={aid: 0.0 for aid in approaches},
        )

        # Metrics history
        self.metrics_history: List[MetricsSnapshot] = []
        self.controller_label: str = "unknown"

        # Phase switch counter (for the current step)
        self._phase_switches: int = 0
        self._total_phase_switches: int = 0

        # Incident state
        self._lane_closures: Dict[str, bool] = {}   # approach_id → closed?
        self._pending_events: List[dict] = []

        # Vehicle counter
        self._vehicle_counter: int = 0

    # -----------------------------------------------------------------------
    # Reset
    # -----------------------------------------------------------------------

    def reset(
        self,
        profile: Optional[str] = None,
        seed: Optional[int] = None,
        n_intersections: Optional[int] = None,
    ) -> None:
        """Full reset to initial conditions."""
        if seed is not None:
            self.seed = seed
        if n_intersections is not None:
            self.n_intersections = n_intersections

        self._rng = random.Random(self.seed)
        self.graph, intersections, approaches = build_grid_network(
            n=self.n_intersections,
            min_green=self.min_green,
            max_green=self.max_green,
            saturation_flow=self.saturation_flow,
        )
        self.node_ids = list(intersections.keys())

        if profile:
            self.demand.set_profile(profile)
        self.demand.reset(seed=self.seed)

        self.state = TrafficState(
            time_s=0,
            intersections=intersections,
            approaches=approaches,
            arrival_buffers={aid: 0.0 for aid in approaches},
        )

        self.metrics_history = []
        self._phase_switches = 0
        self._total_phase_switches = 0
        self._lane_closures = {}
        self._pending_events = []
        self._vehicle_counter = 0

    # -----------------------------------------------------------------------
    # Apply a signal plan (called by controllers before stepping)
    # -----------------------------------------------------------------------

    def apply_signal_plan(self, assignments: Dict[str, int]) -> None:
        """
        Apply a validated signal plan.
        assignments: {node_id → target_phase}
        If an intersection is in clearance or reserved, ignore the assignment.
        """
        s = self.state
        self._phase_switches = 0
        for node_id, target_phase in assignments.items():
            inter = s.intersections.get(node_id)
            if inter is None:
                continue

            # Emergency reservation cannot be overridden
            if inter.is_reserved and inter.reserved_phase is not None:
                target_phase = inter.reserved_phase

            # Skip if already in clearance
            if inter.is_in_clearance:
                continue

            if target_phase != inter.phase and inter.elapsed_green >= inter.min_green:
                # Start yellow transition
                inter.yellow_remaining = self.yellow_s
                inter.all_red_remaining = self.all_red_s
                inter.elapsed_green = 0
                inter.last_switch_time = s.time_s
                self._phase_switches += 1
                self._total_phase_switches += 1
                # Actual phase will flip after yellow+all-red clears (in _tick_signals)
                inter._pending_phase = target_phase  # type: ignore[attr-defined]
            elif target_phase == inter.phase:
                pass  # no action needed

    # -----------------------------------------------------------------------
    # Main step
    # -----------------------------------------------------------------------

    def step(self, n_ticks: int = 1) -> TrafficState:
        """Advance the simulation by n_ticks seconds."""
        for _ in range(n_ticks):
            self._tick()
        return self.state

    def _tick(self) -> None:
        s = self.state
        s.time_s += 1
        t = s.time_s

        # 1. Process signal timers (yellow, all-red)
        self._tick_signals(t)

        # 2. Generate arrivals
        self._tick_arrivals(t)

        # 3. Service queues for green approaches
        self._tick_service()

        # 4. Advance vehicles, update emergency
        self._tick_vehicles(t)

        # 5. Update waiting/starvation timers
        self._tick_waiting()

        # 6. Check max-green forced switches
        self._tick_max_green(t)

    # -----------------------------------------------------------------------
    # Signal timer processing
    # -----------------------------------------------------------------------

    def _tick_signals(self, t: int) -> None:
        for inter in self.state.intersections.values():
            if inter.yellow_remaining > 0:
                inter.yellow_remaining -= 1
                if inter.yellow_remaining == 0:
                    # Move to all-red
                    pass  # all_red countdown starts
            elif inter.all_red_remaining > 0:
                inter.all_red_remaining -= 1
                if inter.all_red_remaining == 0:
                    # Flip to pending phase
                    pending = getattr(inter, "_pending_phase", None)
                    if pending is not None:
                        inter.phase = pending
                        del inter._pending_phase  # type: ignore[attr-defined]
                    inter.elapsed_green = 0
            else:
                # Normal green tick
                if inter.phase in (PHASE_NS, PHASE_EW):
                    inter.elapsed_green += 1

            # Expire reservations
            if inter.is_reserved and t > inter.reserved_until:
                inter.reserved_phase = None
                inter.reserved_until = -1

    # -----------------------------------------------------------------------
    # Arrival generation
    # -----------------------------------------------------------------------

    def _tick_arrivals(self, t: int) -> None:
        s = self.state
        for approach in s.approaches.values():
            closed = self._lane_closures.get(approach.approach_id, False)
            n_arr = self.demand.arrivals(
                approach_id=approach.approach_id,
                movement=approach.movement,
                time_s=t,
                lane_closure=closed,
            )
            approach.queue += n_arr
            approach.total_arrivals += n_arr

            types_dist = ["car", "car", "suv", "taxi", "bus", "truck", "ev"]
            for _ in range(n_arr):
                self._vehicle_counter += 1
                vid = f"V{self._vehicle_counter}"
                origin = approach.intersection_id
                destinations = [n for n in self.node_ids if n != origin]
                dest = self._rng.choice(destinations) if destinations else origin
                v_type = str(self._rng.choice(types_dist))
                pax = int(self._rng.randint(12, 45)) if v_type == "bus" else (2 if v_type == "taxi" else 1)

                # Determine route across intersection points
                route = find_shortest_path(self.graph, origin, dest)
                if not route or len(route) < 2:
                    route = [origin, dest]
                from_n = route[0]
                to_n = route[1] if len(route) > 1 else dest
                mvt = get_edge_direction(self.graph, from_n, to_n) or approach.movement

                v = Vehicle(
                    vehicle_id=vid,
                    origin=origin,
                    destination=dest,
                    route=route,
                    route_index=0,
                    arrival_time=t,
                    current_node=origin,
                    vehicle_type=v_type,
                    speed_kmh=42.0,
                    state_expression="cruising",
                    passengers=pax,
                    from_node=from_n,
                    to_node=to_n,
                    progress=0.0,
                    is_stopped=False,
                    movement=mvt,
                )
                s.vehicles[vid] = v

    # -----------------------------------------------------------------------
    # Queue service
    # -----------------------------------------------------------------------

    def _tick_service(self) -> None:
        s = self.state
        for approach in s.approaches.values():
            inter = s.intersections.get(approach.intersection_id)
            if inter is None:
                continue
            # Only serve if intersection is in compatible phase and not in clearance
            if inter.is_in_clearance:
                continue
            if inter.phase != approach.compatible_phase:
                continue
            if self._lane_closures.get(approach.approach_id, False):
                # Lane closure: halve service rate
                rate = approach.max_service_rate * 0.5
            else:
                rate = approach.max_service_rate

            served = min(approach.queue, rate)
            approach.queue = max(0.0, approach.queue - served)
            approach.total_served += int(served)
            approach.starvation_timer = 0

    # -----------------------------------------------------------------------
    # Vehicle advancement
    # -----------------------------------------------------------------------

    def _tick_vehicles(self, t: int) -> None:
        s = self.state
        completed = []

        # Group active vehicles by road corridor link (from_node, to_node)
        link_vehicles: Dict[Tuple[str, str], List[Vehicle]] = {}
        for v in s.vehicles.values():
            if not v.completed:
                link_vehicles.setdefault((v.from_node, v.to_node), []).append(v)

        for (from_n, to_n), v_list in link_vehicles.items():
            # Sort vehicles from front of line (highest progress) to rear (lowest progress)
            v_list.sort(key=lambda x: x.progress, reverse=True)

            inter = s.intersections.get(to_n)
            stop_line = 0.72  # stop threshold before intersection

            for rank, v in enumerate(v_list):
                # Safe queue stop slot behind the vehicle in front (0.72, 0.58, 0.44, ...)
                queue_slot = max(0.12, stop_line - rank * 0.14)

                # Check if signal is green for this vehicle's direction
                is_green = False
                if inter:
                    if v.emergency:
                        is_green = True
                    elif not inter.is_in_clearance:
                        req_phase = PHASE_NS if v.movement == "NS" else PHASE_EW
                        is_green = (inter.phase == req_phase)

                # Stop if approaching/at stop line and signal is NOT green
                if v.progress >= queue_slot and not is_green:
                    if not v.is_stopped:
                        v.stops += 1
                    v.progress = queue_slot
                    v.is_stopped = True
                    v.speed_kmh = 0.0
                    v.state_expression = "idling"
                    v.waiting_time += 1
                else:
                    # Signal is GREEN or road ahead is clear: advance through points!
                    v.is_stopped = False
                    step = 0.22 if v.emergency else 0.15
                    v.progress = min(1.05, v.progress + step)
                    v.speed_kmh = 75.0 if v.emergency else (48.0 if v.progress < stop_line else 32.0)
                    v.state_expression = "emergency_rush" if v.emergency else ("accelerating" if v.speed_kmh < 40 else "cruising")

                # If vehicle reached and crossed the intersection (progress >= 1.0) and signal is green:
                if v.progress >= 1.0 and is_green:
                    v.route_index += 1
                    if v.route and v.route_index < len(v.route) - 1:
                        # Advance to next road link along the route
                        v.from_node = v.route[v.route_index]
                        v.to_node = v.route[v.route_index + 1]
                        v.current_node = v.from_node
                        v.movement = get_edge_direction(self.graph, v.from_node, v.to_node) or "NS"
                        v.progress = 0.0
                        v.is_stopped = False
                    else:
                        # Reached journey destination!
                        v.departure_time = t
                        completed.append(v.vehicle_id)

            # Clean up vehicles exceeding max trip timeout
            for v in v_list:
                if t - v.arrival_time >= 150 and v.vehicle_id not in completed:
                    v.departure_time = t
                    completed.append(v.vehicle_id)

        for vid in completed:
            if vid in s.vehicles:
                v = s.vehicles.pop(vid)
                s.completed_vehicles.append(v)

        # Advance emergency vehicle
        if s.emergency and not s.emergency.cleared:
            self._advance_emergency(t)

    def _advance_emergency(self, t: int) -> None:
        em = self.state.emergency
        assert em is not None
        # Advance one step every 10 seconds
        if t % 10 == 0 and em.current_index < len(em.route) - 1:
            em.current_index += 1
        if em.current_index >= len(em.route) - 1:
            em.cleared = True
            em.clearance_time = t
            # Release reservations
            for node_id in em.route:
                inter = self.state.intersections.get(node_id)
                if inter:
                    inter.reserved_phase = None
                    inter.reserved_until = -1

    # -----------------------------------------------------------------------
    # Waiting timers
    # -----------------------------------------------------------------------

    def _tick_waiting(self) -> None:
        for approach in self.state.approaches.values():
            inter = self.state.intersections.get(approach.intersection_id)
            if inter and inter.phase != approach.compatible_phase:
                # Vehicles in this approach are waiting
                waiting_vehicles = int(approach.queue)
                approach.cumulative_wait += waiting_vehicles
                approach.starvation_timer += 1

    # -----------------------------------------------------------------------
    # Max-green forced switch
    # -----------------------------------------------------------------------

    def _tick_max_green(self, t: int) -> None:
        for inter in self.state.intersections.values():
            if inter.elapsed_green >= inter.max_green and not inter.is_in_clearance:
                if inter.is_reserved:
                    continue
                # Force switch to other phase
                other = PHASE_EW if inter.phase == PHASE_NS else PHASE_NS
                inter.yellow_remaining = self.yellow_s
                inter.all_red_remaining = self.all_red_s
                inter._pending_phase = other  # type: ignore[attr-defined]
                inter.elapsed_green = 0
                self._total_phase_switches += 1

    # -----------------------------------------------------------------------
    # Incident application
    # -----------------------------------------------------------------------

    def apply_lane_closure(self, approach_id: str, closed: bool = True) -> None:
        self._lane_closures[approach_id] = closed

    def apply_demand_spike(
        self, approach_id: str, extra_rate: float, duration_s: int
    ) -> None:
        self.demand.apply_spike(approach_id, extra_rate, duration_s)

    def clear_demand_spike(self, approach_id: str) -> None:
        self.demand.clear_spike(approach_id)

    # -----------------------------------------------------------------------
    # Metrics snapshot
    # -----------------------------------------------------------------------

    def record_metrics(self, controller: str, qaoa_energy: Optional[float] = None,
                       qaoa_gap: Optional[float] = None) -> MetricsSnapshot:
        s = self.state
        queues = [a.queue for a in s.approaches.values()]
        mean_q = sum(queues) / len(queues) if queues else 0.0
        max_q = max(queues) if queues else 0.0
        total_q = sum(queues)

        completed = s.completed_vehicles
        n_comp = len(completed)
        mean_wait = (
            sum(v.waiting_time for v in completed) / n_comp if n_comp > 0 else 0.0
        )
        mean_tt = (
            sum((v.travel_time or 0) for v in completed) / n_comp if n_comp > 0 else 0.0
        )

        # Estimated idling fuel & CO2 (model-based estimate, relative comparison only)
        idle_vehicles = sum(int(a.queue) for a in s.approaches.values())
        fuel = idle_vehicles * self.idle_fuel
        co2 = fuel * self.co2_per_litre

        snap = MetricsSnapshot(
            time_s=s.time_s,
            controller=controller,
            mean_queue=mean_q,
            max_queue=max_q,
            total_queue=total_q,
            throughput_total=n_comp,
            mean_wait=mean_wait,
            mean_travel_time=mean_tt,
            phase_switches=self._total_phase_switches,
            estimated_fuel_L=fuel,
            estimated_co2_kg=co2,
            qaoa_energy=qaoa_energy,
            qaoa_gap_pct=qaoa_gap,
            emergency_active=s.emergency is not None and not s.emergency.cleared,
        )
        self.metrics_history.append(snap)
        return snap
