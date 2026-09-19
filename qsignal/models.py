"""
Core data models for Q-Signal.
Kept independent of Streamlit so the simulator and controllers can be
tested without launching the UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Signal phase constants
# ---------------------------------------------------------------------------
PHASE_NS = 0   # North-South movement is green
PHASE_EW = 1   # East-West movement is green
PHASE_YELLOW = 2
PHASE_ALL_RED = 3

PHASE_NAMES = {PHASE_NS: "NS", PHASE_EW: "EW", PHASE_YELLOW: "Yellow", PHASE_ALL_RED: "All-Red"}

# Movement → compatible phase mapping
MOVEMENT_PHASE = {"NS": PHASE_NS, "EW": PHASE_EW}
PHASE_MOVEMENT = {PHASE_NS: "NS", PHASE_EW: "EW"}

# Conflicting movement pairs (cannot both be green simultaneously)
CONFLICT_PAIRS = {("NS", "EW"), ("EW", "NS")}


# ---------------------------------------------------------------------------
# Core dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Intersection:
    """A signalized intersection node in the network."""
    node_id: str
    position: Tuple[float, float]        # (lat, lon) or schematic (x, y)
    phase: int = PHASE_NS                # current active phase
    elapsed_green: int = 0               # seconds in current green phase
    min_green: int = 10
    max_green: int = 40
    yellow_remaining: int = 0            # seconds of yellow clearance remaining
    all_red_remaining: int = 0           # seconds of all-red remaining
    reserved_phase: Optional[int] = None # emergency reservation
    reserved_until: int = -1             # simulation time when reservation expires
    last_switch_time: int = 0            # simulation time of last phase switch

    @property
    def is_in_clearance(self) -> bool:
        return self.yellow_remaining > 0 or self.all_red_remaining > 0

    @property
    def is_reserved(self) -> bool:
        return self.reserved_phase is not None

    def can_switch(self, current_time: int) -> bool:
        """True if the intersection is allowed to change phase."""
        return (
            not self.is_in_clearance
            and self.elapsed_green >= self.min_green
        )


@dataclass
class Approach:
    """A directed lane group entering an intersection from one direction."""
    approach_id: str
    intersection_id: str
    movement: str                         # "NS" or "EW"
    lanes: int = 1
    saturation_flow: float = 0.5          # vehicles/second/lane at full green
    queue: float = 0.0                    # current vehicle queue
    total_arrivals: int = 0
    total_served: int = 0
    cumulative_wait: float = 0.0          # vehicle-seconds of waiting
    starvation_timer: int = 0             # seconds since last served

    @property
    def compatible_phase(self) -> int:
        return MOVEMENT_PHASE[self.movement]

    @property
    def max_service_rate(self) -> float:
        return self.saturation_flow * self.lanes


@dataclass
class Vehicle:
    """A vehicle travelling through the network."""
    vehicle_id: str
    origin: str
    destination: str
    route: List[str]                      # ordered list of node_ids
    emergency: bool = False
    route_index: int = 0                  # current position in route
    arrival_time: int = 0                 # simulation time vehicle entered network
    departure_time: Optional[int] = None  # simulation time vehicle exited
    current_node: str = ""
    stops: int = 0                        # number of times vehicle stopped
    waiting_time: int = 0                 # total seconds spent waiting
    vehicle_type: str = "car"             # "car", "suv", "taxi", "bus", "truck", "ev", "ambulance"
    speed_kmh: float = 42.0               # dynamic speed in km/h
    state_expression: str = "cruising"    # "cruising", "idling", "accelerating", "braking", "emergency_rush"
    passengers: int = 1
    from_node: str = ""                   # current origin of road segment
    to_node: str = ""                     # current destination of road segment
    progress: float = 0.0                 # position along link (0.0 to 1.0)
    is_stopped: bool = False              # whether stopped at red signal or queue
    movement: str = "NS"                  # "NS" or "EW"

    @property
    def icon(self) -> str:
        if self.emergency or self.vehicle_type == "ambulance":
            return "🚑"
        icons = {
            "car": "🚗",
            "suv": "🚙",
            "taxi": "🚕",
            "bus": "🚌",
            "truck": "🚚",
            "ev": "⚡🚙",
        }
        return icons.get(self.vehicle_type, "🚗")

    @property
    def expression_label(self) -> str:
        if self.emergency:
            return "🚨 Code 3 Priority Wave"
        if self.is_stopped:
            return f"🛑 Stopped at Red Light ({self.waiting_time}s)"
        if self.state_expression == "idling":
            return f"🔴 Idling in Queue ({self.waiting_time}s wait)"
        if self.state_expression == "braking":
            return "🟡 Decelerating (Signal Red)"
        if self.state_expression == "accelerating":
            return f"🟢 Clearing Intersection ({self.speed_kmh:.0f} km/h)"
        return f"🟢 Moving to {self.to_node or self.destination} ({self.speed_kmh:.0f} km/h)"

    @property
    def completed(self) -> bool:
        return self.departure_time is not None

    @property
    def travel_time(self) -> Optional[int]:
        if self.departure_time is not None:
            return self.departure_time - self.arrival_time
        return None


@dataclass
class SignalPlan:
    """A proposed signal assignment for one decision interval."""
    assignments: Dict[str, int]           # node_id → phase (PHASE_NS or PHASE_EW)
    objective_value: float = 0.0
    feasible: bool = True
    solver: str = "unknown"               # "fixed", "pressure", "qaoa", "exact", "fallback"
    qaoa_energy: Optional[float] = None
    qaoa_gap_pct: Optional[float] = None
    bits: Optional[List[int]] = None


@dataclass
class EmergencyRequest:
    """An active emergency vehicle and its green corridor."""
    vehicle_id: str
    origin: str
    destination: str
    route: List[str]                      # node_ids from origin to destination
    route_phases: List[int]               # compatible phase per node
    requested_at: int                     # simulation time of injection
    current_index: int = 0               # which route segment vehicle is at
    cleared: bool = False
    clearance_time: Optional[int] = None  # simulation time when corridor released

    @property
    def next_nodes(self) -> List[str]:
        """Nodes still ahead of the emergency vehicle."""
        return self.route[self.current_index:]


@dataclass
class TrafficState:
    """Complete snapshot of the traffic network at a given simulation time."""
    time_s: int
    intersections: Dict[str, Intersection]
    approaches: Dict[str, Approach]
    vehicles: Dict[str, Vehicle] = field(default_factory=dict)
    incidents: List[dict] = field(default_factory=list)
    emergency: Optional[EmergencyRequest] = None
    arrival_buffers: Dict[str, float] = field(default_factory=dict)
    completed_vehicles: List[Vehicle] = field(default_factory=list)


@dataclass
class MetricsSnapshot:
    """Metrics recorded at a single simulation step."""
    time_s: int
    controller: str
    mean_queue: float
    max_queue: float
    total_queue: float
    throughput_total: int               # cumulative vehicles completed
    mean_wait: float
    mean_travel_time: float
    phase_switches: int
    estimated_fuel_L: float             # documented estimate
    estimated_co2_kg: float             # documented estimate
    qaoa_energy: Optional[float] = None
    qaoa_gap_pct: Optional[float] = None
    emergency_active: bool = False
