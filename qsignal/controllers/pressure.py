"""
Pressure-based (actuated) signal controller for Q-Signal.
Gives green to the movement with the highest weighted queue pressure.
Includes hysteresis to prevent rapid oscillation.
"""

from __future__ import annotations

from typing import Dict

from qsignal.models import (
    TrafficState,
    SignalPlan,
    Approach,
    Intersection,
    PHASE_NS,
    PHASE_EW,
    PHASE_MOVEMENT,
)


class PressureController:
    """
    Rule-based pressure controller.

    Pressure = queue_size × saturation_flow × lanes.
    When pressure difference between phases falls below hysteresis_threshold,
    keep the current phase to avoid unnecessary switching.
    """

    def __init__(
        self,
        hysteresis_threshold: float = 0.5,
        starvation_limit_s: int = 60,
    ):
        self.hysteresis_threshold = hysteresis_threshold
        self.starvation_limit_s = starvation_limit_s

    def reset(self) -> None:
        pass  # stateless beyond what the simulator tracks

    def compute_action(self, state: TrafficState) -> SignalPlan:
        """Return a pressure-based phase assignment."""
        assignments: Dict[str, int] = {}

        # Group approaches by intersection
        by_node: Dict[str, Dict[str, Approach]] = {}
        for approach in state.approaches.values():
            by_node.setdefault(approach.intersection_id, {})[approach.movement] = approach

        for node_id, inter in state.intersections.items():
            ap = by_node.get(node_id, {})
            ns_app = ap.get("NS")
            ew_app = ap.get("EW")

            p_ns = self._pressure(ns_app) if ns_app else 0.0
            p_ew = self._pressure(ew_app) if ew_app else 0.0

            # Starvation override: if one movement hasn't been served for too long,
            # force its phase regardless of pressure balance
            if ns_app and ns_app.starvation_timer >= self.starvation_limit_s:
                assignments[node_id] = PHASE_NS
                continue
            if ew_app and ew_app.starvation_timer >= self.starvation_limit_s:
                assignments[node_id] = PHASE_EW
                continue

            # Hysteresis: only switch if pressure difference exceeds threshold
            current_phase = inter.phase
            diff = p_ns - p_ew  # positive → NS has more pressure

            if current_phase == PHASE_NS:
                # Switch to EW only if EW pressure is significantly higher
                if diff < -self.hysteresis_threshold:
                    assignments[node_id] = PHASE_EW
                else:
                    assignments[node_id] = PHASE_NS
            else:
                # Switch to NS only if NS pressure is significantly higher
                if diff > self.hysteresis_threshold:
                    assignments[node_id] = PHASE_NS
                else:
                    assignments[node_id] = PHASE_EW

        return SignalPlan(
            assignments=assignments,
            objective_value=sum(
                self._pressure(a)
                for a in state.approaches.values()
            ),
            feasible=True,
            solver="pressure",
        )

    def _pressure(self, approach: Approach) -> float:
        """Pressure = queue × max_service_rate."""
        return approach.queue * approach.max_service_rate
