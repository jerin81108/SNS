"""
Event types and injection helpers for Q-Signal.
Incidents: congestion spike, lane closure, accident, emergency vehicle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from qsignal.simulator import TrafficSimulator


@dataclass
class Event:
    """A scheduled or on-demand event."""
    event_type: str          # "spike", "closure", "accident", "emergency"
    target_id: str           # approach_id or node_id
    start_time: int
    duration_s: int = 60
    extra_data: dict = field(default_factory=dict)
    active: bool = False
    cleared: bool = False


class EventManager:
    """
    Manages incident events and applies them to the simulator state.
    """

    def __init__(self, sim: "TrafficSimulator"):
        self.sim = sim
        self.events: List[Event] = []

    def inject_congestion_spike(
        self,
        approach_id: str,
        extra_rate: float = 0.40,
        duration_s: int = 60,
    ) -> Event:
        """
        Temporarily boost arrivals on one approach.
        """
        ev = Event(
            event_type="spike",
            target_id=approach_id,
            start_time=self.sim.state.time_s,
            duration_s=duration_s,
            extra_data={"extra_rate": extra_rate},
        )
        self.events.append(ev)
        self.sim.apply_demand_spike(approach_id, extra_rate, duration_s)
        ev.active = True
        return ev

    def inject_lane_closure(
        self,
        approach_id: str,
        duration_s: int = 120,
    ) -> Event:
        """
        Close a lane on one approach, halving its service rate.
        """
        ev = Event(
            event_type="closure",
            target_id=approach_id,
            start_time=self.sim.state.time_s,
            duration_s=duration_s,
        )
        self.events.append(ev)
        self.sim.apply_lane_closure(approach_id, closed=True)
        ev.active = True
        return ev

    def inject_accident(
        self,
        approach_id: str,
        duration_s: int = 90,
    ) -> Event:
        """
        Simulate an accident: lane closure + small demand spike.
        """
        ev = Event(
            event_type="accident",
            target_id=approach_id,
            start_time=self.sim.state.time_s,
            duration_s=duration_s,
        )
        self.events.append(ev)
        self.sim.apply_lane_closure(approach_id, closed=True)
        self.sim.apply_demand_spike(approach_id, extra_rate=0.20, duration_s=duration_s)
        ev.active = True
        return ev

    def inject_emergency_vehicle(
        self,
        origin: str,
        destination: str,
        clearance_buffer_s: int = 15,
        advance_reservations: int = 2,
    ) -> Optional[Event]:
        """
        Inject an emergency vehicle and activate the green corridor.
        Returns None if no route exists.
        """
        from qsignal.emergency import EmergencyCorridorManager
        em_mgr = EmergencyCorridorManager(self.sim, clearance_buffer_s, advance_reservations)
        em = em_mgr.activate(origin, destination)
        if em is None:
            return None

        ev = Event(
            event_type="emergency",
            target_id=origin,
            start_time=self.sim.state.time_s,
            duration_s=300,
            extra_data={"route": em.route, "em": em},
        )
        self.events.append(ev)
        ev.active = True
        return ev

    def tick(self) -> None:
        """
        Check if any active events should expire and clear them.
        Called every simulation tick.
        """
        t = self.sim.state.time_s
        for ev in self.events:
            if ev.active and not ev.cleared:
                elapsed = t - ev.start_time
                if elapsed >= ev.duration_s:
                    self._clear_event(ev)
                    ev.cleared = True
                    ev.active = False

    def _clear_event(self, ev: Event) -> None:
        if ev.event_type == "spike":
            self.sim.clear_demand_spike(ev.target_id)
        elif ev.event_type in ("closure", "accident"):
            self.sim.apply_lane_closure(ev.target_id, closed=False)
            if ev.event_type == "accident":
                self.sim.clear_demand_spike(ev.target_id)
        elif ev.event_type == "emergency":
            pass  # Emergency corridor clears itself

    def active_events(self) -> List[Event]:
        return [e for e in self.events if e.active]
