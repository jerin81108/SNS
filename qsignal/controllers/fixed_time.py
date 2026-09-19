"""
Fixed-time signal controller for Q-Signal.
Alternates phases on a fixed cycle regardless of traffic conditions.
"""

from __future__ import annotations

from typing import Dict

from qsignal.models import TrafficState, SignalPlan, PHASE_NS, PHASE_EW


class FixedTimeController:
    """
    Alternates all intersections between NS and EW phases on a fixed cycle.
    Uses the same min_green, yellow, and all-red durations as the optimized controller.
    """

    def __init__(self, green_duration: int = 20):
        self.green_duration = green_duration
        self._cycle_counters: Dict[str, int] = {}

    def reset(self) -> None:
        self._cycle_counters = {}

    def compute_action(self, state: TrafficState) -> SignalPlan:
        """
        Return the fixed-time phase assignment.
        Each intersection alternates every green_duration seconds (in simulation time).
        """
        assignments: Dict[str, int] = {}

        for node_id, inter in state.intersections.items():
            # Determine phase based on elapsed time modulo cycle
            cycle_s = 2 * (self.green_duration + 3 + 2)  # green + yellow + all-red × 2
            half = cycle_s // 2
            time_in_cycle = state.time_s % cycle_s
            target = PHASE_NS if time_in_cycle < half else PHASE_EW
            assignments[node_id] = target

        return SignalPlan(
            assignments=assignments,
            objective_value=0.0,
            feasible=True,
            solver="fixed",
        )
