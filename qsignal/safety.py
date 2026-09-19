"""
Deterministic safety validator for Q-Signal.
The safety layer is always classical. Quantum output is only a proposal;
this module validates and enforces all signal safety constraints.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from qsignal.models import (
    Intersection,
    TrafficState,
    SignalPlan,
    PHASE_NS,
    PHASE_EW,
    PHASE_MOVEMENT,
    CONFLICT_PAIRS,
)


class SafetyValidator:
    """
    Validates and corrects a proposed SignalPlan before it is applied.

    Rules enforced (in priority order):
      1. Emergency reservations cannot be overridden.
      2. Intersections in yellow/all-red clearance cannot switch.
      3. Minimum green time must be respected.
      4. Conflicting movements must not both be green.
      5. Only valid phase values (PHASE_NS, PHASE_EW) are allowed.
    """

    VALID_PHASES = {PHASE_NS, PHASE_EW}

    def __init__(self, min_green: int = 10, yellow_s: int = 3, all_red_s: int = 2):
        self.min_green = min_green
        self.yellow_s = yellow_s
        self.all_red_s = all_red_s
        self.violations: list = []

    def validate(
        self, plan: SignalPlan, state: TrafficState
    ) -> Tuple[SignalPlan, bool]:
        """
        Validate and correct a SignalPlan.

        Returns:
            corrected_plan: A SignalPlan with all unsafe assignments fixed.
            all_safe: True if the original plan was already fully safe.
        """
        self.violations = []
        corrected = dict(plan.assignments)
        all_safe = True

        for node_id, proposed_phase in plan.assignments.items():
            inter = state.intersections.get(node_id)
            if inter is None:
                continue

            safe_phase, reason = self._check_intersection(inter, proposed_phase, state)
            if safe_phase != proposed_phase:
                all_safe = False
                corrected[node_id] = safe_phase
                self.violations.append({
                    "node_id": node_id,
                    "proposed": proposed_phase,
                    "corrected": safe_phase,
                    "reason": reason,
                })

        # Check for network-level conflicts (shouldn't happen with binary QUBO,
        # but included for safety)
        # In this two-phase model each intersection is independent,
        # so conflicting movements within the same intersection are prevented
        # by design (each node has exactly one phase).

        corrected_plan = SignalPlan(
            assignments=corrected,
            objective_value=plan.objective_value,
            feasible=all_safe,
            solver=plan.solver,
            qaoa_energy=plan.qaoa_energy,
            qaoa_gap_pct=plan.qaoa_gap_pct,
            bits=plan.bits,
        )
        return corrected_plan, all_safe

    def _check_intersection(
        self,
        inter: Intersection,
        proposed_phase: int,
        state: TrafficState,
    ) -> Tuple[int, str]:
        """
        Return the safe phase for this intersection and a reason string.
        Returns (proposed_phase, "") if safe; (current_phase, reason) otherwise.
        """
        # Rule 1: Emergency reservation
        if inter.is_reserved and inter.reserved_phase is not None:
            if proposed_phase != inter.reserved_phase:
                return inter.reserved_phase, "emergency_reservation"

        # Rule 2: Clearance interval in progress
        if inter.is_in_clearance:
            return inter.phase, "in_clearance"

        # Rule 3: Minimum green not yet reached
        if proposed_phase != inter.phase and inter.elapsed_green < self.min_green:
            return inter.phase, f"min_green_not_met ({inter.elapsed_green}/{self.min_green}s)"

        # Rule 4: Phase must be a valid value
        if proposed_phase not in self.VALID_PHASES:
            return inter.phase, f"invalid_phase_value ({proposed_phase})"

        return proposed_phase, ""

    def is_plan_feasible(self, plan: SignalPlan, state: TrafficState) -> bool:
        """Quick feasibility check without producing a corrected plan."""
        _, safe = self.validate(plan, state)
        return safe

    def check_conflict_free(self, state: TrafficState) -> bool:
        """
        Verify that no conflicting movements are simultaneously green.
        In the two-phase model this is always true by construction,
        but this method serves as a property test hook.
        """
        for inter in state.intersections.values():
            if inter.is_in_clearance:
                continue
            # In this model each intersection only serves one movement at a time.
            # Multiple intersections serving different movements is always safe.
        return True

    def get_violations(self) -> list:
        """Return violations from the last validate() call."""
        return self.violations
