"""
Emergency green corridor module for Q-Signal.
Handles route finding, phase reservation, and corridor restoration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from qsignal.models import EmergencyRequest, PHASE_NS
from qsignal.network import find_shortest_path, get_route_phases

if TYPE_CHECKING:
    from qsignal.simulator import TrafficSimulator


class EmergencyCorridorManager:
    """
    Manages emergency green corridors.

    Priority order:
      safety validation > emergency corridor > min-green rules > normal optimizer > fallback
    """

    def __init__(
        self,
        sim: "TrafficSimulator",
        clearance_buffer_s: int = 15,
        advance_reservations: int = 2,
    ):
        self.sim = sim
        self.clearance_buffer_s = clearance_buffer_s
        self.advance_reservations = advance_reservations

    def activate(
        self, origin: str, destination: str
    ) -> Optional[EmergencyRequest]:
        """
        Find a route, create an EmergencyRequest, and reserve phases.
        Returns None if no path exists or origin/destination are invalid.
        """
        state = self.sim.state
        graph = self.sim.graph

        if origin not in state.intersections or destination not in state.intersections:
            return None
        if origin == destination:
            return None

        route = find_shortest_path(graph, origin, destination)
        if route is None or len(route) < 2:
            return None

        route_phases = get_route_phases(graph, route)

        em = EmergencyRequest(
            vehicle_id="EMV-1",
            origin=origin,
            destination=destination,
            route=route,
            route_phases=route_phases,
            requested_at=state.time_s,
        )
        state.emergency = em

        # Reserve the first advance_reservations intersections
        self._update_reservations(em)
        return em

    def _update_reservations(self, em: EmergencyRequest) -> None:
        """
        Reserve phases for the next advance_reservations intersections
        ahead of the emergency vehicle's current position.
        """
        state = self.sim.state
        t = state.time_s
        start = em.current_index
        end = min(start + self.advance_reservations + 1, len(em.route))

        for i in range(start, end):
            node_id = em.route[i]
            req_phase = em.route_phases[i] if i < len(em.route_phases) else PHASE_NS
            inter = state.intersections.get(node_id)
            if inter is None:
                continue

            # Set reservation
            inter.reserved_phase = req_phase
            inter.reserved_until = t + 60 + self.clearance_buffer_s

            # Force the intersection to the required phase if possible
            if not inter.is_in_clearance and inter.phase != req_phase:
                if inter.elapsed_green >= inter.min_green:
                    inter.yellow_remaining = self.sim.yellow_s
                    inter.all_red_remaining = self.sim.all_red_s
                    inter._pending_phase = req_phase  # type: ignore[attr-defined]
                    inter.elapsed_green = 0

    def tick(self) -> None:
        """
        Called every decision interval to advance reservations
        as the emergency vehicle progresses.
        """
        state = self.sim.state
        em = state.emergency
        if em is None or em.cleared:
            return
        self._update_reservations(em)

    def release(self) -> None:
        """Manually release the emergency corridor."""
        state = self.sim.state
        if state.emergency is None:
            return
        em = state.emergency
        for node_id in em.route:
            inter = state.intersections.get(node_id)
            if inter:
                inter.reserved_phase = None
                inter.reserved_until = -1
        em.cleared = True
        em.clearance_time = state.time_s

    def get_status(self) -> dict:
        """Return a status dict for dashboard display."""
        em = self.sim.state.emergency
        if em is None:
            return {"active": False}
        elapsed = self.sim.state.time_s - em.requested_at
        return {
            "active": not em.cleared,
            "route": em.route,
            "current_index": em.current_index,
            "origin": em.origin,
            "destination": em.destination,
            "elapsed_s": elapsed,
            "clearance_time": em.clearance_time,
            "reserved_nodes": [
                n for n in em.route
                if self.sim.state.intersections.get(n) and
                   self.sim.state.intersections[n].is_reserved
            ],
        }
