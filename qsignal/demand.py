"""
Demand generation for Q-Signal.
Provides deterministic fractional accumulation (easy to debug)
and optional Poisson sampling for more realistic variation.
"""

from __future__ import annotations

import random
from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Built-in demand profiles (vehicles per second per movement)
# ---------------------------------------------------------------------------

DEMAND_PROFILES: Dict[str, Dict[str, float]] = {
    "normal": {"NS": 0.20, "EW": 0.15},
    "morning_peak": {"NS": 0.45, "EW": 0.25},
    "evening_peak": {"NS": 0.25, "EW": 0.50},
    "heavy": {"NS": 0.55, "EW": 0.55},
    "low": {"NS": 0.08, "EW": 0.06},
}

# Time-varying multipliers: list of (start_s, end_s, multiplier)
TIME_VARYING_SCHEDULE = [
    (0, 180, 1.0),
    (180, 360, 1.5),
    (360, 540, 1.0),
    (540, 720, 2.0),
    (720, 900, 1.0),
]


def get_time_multiplier(time_s: int) -> float:
    """Return demand multiplier for the current simulation time."""
    for start, end, mult in TIME_VARYING_SCHEDULE:
        if start <= (time_s % 900) < end:
            return mult
    return 1.0


class DemandGenerator:
    """
    Generates vehicle arrivals for each approach using deterministic
    fractional accumulation. Seeded for reproducibility.
    """

    def __init__(
        self,
        profile: str = "normal",
        seed: int = 42,
        noise_sigma: float = 0.0,
        time_varying: bool = True,
    ):
        self.profile_name = profile
        self.base_rates = DEMAND_PROFILES.get(profile, DEMAND_PROFILES["normal"]).copy()
        self.noise_sigma = noise_sigma
        self.time_varying = time_varying
        self._rng = random.Random(seed)
        self._buffers: Dict[str, float] = {}
        self._spike_overrides: Dict[str, float] = {}  # approach_id → extra rate

    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> None:
        """Reset buffers and optionally re-seed the RNG."""
        self._buffers = {}
        self._spike_overrides = {}
        if seed is not None:
            self._rng = random.Random(seed)

    def set_profile(self, profile: str) -> None:
        self.profile_name = profile
        self.base_rates = DEMAND_PROFILES.get(profile, DEMAND_PROFILES["normal"]).copy()

    def apply_spike(self, approach_id: str, extra_rate: float, duration_s: int) -> None:
        """
        Temporarily boost demand on one approach.
        The spike expires after duration_s seconds; caller must call clear_spike.
        """
        self._spike_overrides[approach_id] = extra_rate

    def clear_spike(self, approach_id: str) -> None:
        self._spike_overrides.pop(approach_id, None)

    # ------------------------------------------------------------------
    def arrivals(
        self,
        approach_id: str,
        movement: str,
        time_s: int,
        lane_closure: bool = False,
    ) -> int:
        """
        Compute integer arrivals for this approach at the current tick.
        Uses fractional accumulation: leftover fractions carry to the next tick.
        """
        base = self.base_rates.get(movement, 0.15)

        # Time-of-day multiplier
        if self.time_varying:
            base *= get_time_multiplier(time_s)

        # Demand spike override
        base += self._spike_overrides.get(approach_id, 0.0)

        # Lane closure reduces effective supply, not demand (vehicles still arrive,
        # they just can't be served as fast). We model it as 50% extra arrivals.
        if lane_closure:
            base *= 1.5

        # Optional Gaussian noise
        if self.noise_sigma > 0:
            noise = self._rng.gauss(0, self.noise_sigma * base)
            base = max(0.0, base + noise)

        # Fractional accumulation
        buf = self._buffers.get(approach_id, 0.0) + base
        new_vehicles = int(buf)
        self._buffers[approach_id] = buf - new_vehicles
        return new_vehicles

    def get_base_rate(self, movement: str) -> float:
        return self.base_rates.get(movement, 0.15)
