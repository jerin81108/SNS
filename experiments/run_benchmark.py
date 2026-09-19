"""
Benchmark runner for Q-Signal.
Runs all controllers × demand profiles × seeds and saves results to CSV.

Usage:
    python experiments/run_benchmark.py --duration 600 --seeds 5
    python experiments/run_benchmark.py --duration 1200 --seeds 10 --output data/experiments/results.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from itertools import product
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml

from qsignal.simulator import TrafficSimulator
from qsignal.events import EventManager
from qsignal.controllers.hybrid_qaoa import solve_action
from qsignal.safety import SafetyValidator
from qsignal.metrics import snapshots_to_dataframe


CONTROLLERS = ["fixed", "pressure", "qaoa"]
DEMAND_PROFILES = ["normal", "morning_peak", "evening_peak"]
INCIDENTS = ["none"]  # extend with "spike", "closure" if desired

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_single(
    controller: str,
    profile: str,
    seed: int,
    duration_s: int,
    n_intersections: int,
    cfg: dict,
    incident: str = "none",
    verbose: bool = False,
) -> List[dict]:
    """Run one simulation with one controller and return metrics rows."""
    sim_cfg = {
        "min_green_s": cfg["signal"]["min_green_s"],
        "max_green_s": cfg["signal"]["max_green_s"],
        "yellow_s": cfg["signal"]["yellow_s"],
        "all_red_s": cfg["signal"]["all_red_s"],
        "saturation_flow": cfg["network"]["saturation_flow"],
        "idle_fuel_L_per_s": cfg["metrics"]["idle_fuel_L_per_s"],
        "co2_per_litre": cfg["metrics"]["co2_per_litre"],
    }
    sim = TrafficSimulator(
        n_intersections=n_intersections,
        profile=profile,
        seed=seed,
        config=sim_cfg,
    )
    events = EventManager(sim)
    validator = SafetyValidator(
        min_green=sim_cfg["min_green_s"],
        yellow_s=sim_cfg["yellow_s"],
        all_red_s=sim_cfg["all_red_s"],
    )

    qaoa_cfg = {
        **cfg.get("qubo", {}),
        "shots": cfg["qaoa"]["shots"],
        "depth": cfg["qaoa"]["depth"],
        "time_budget_s": cfg["qaoa"]["time_budget_s"],
        "fallback": cfg["qaoa"]["fallback"],
    }

    decision_interval = cfg["simulation"]["decision_interval_s"]
    ticks = 0

    # Inject incident at t=120
    incident_injected = False

    while sim.state.time_s < duration_s:
        t = sim.state.time_s

        # Inject incident at t=120
        if incident == "spike" and t >= 120 and not incident_injected:
            first_approach = list(sim.state.approaches.keys())[0]
            events.inject_congestion_spike(first_approach, extra_rate=0.4, duration_s=60)
            incident_injected = True

        if t % decision_interval == 0:
            plan = solve_action(sim.state, mode=controller, config=qaoa_cfg, graph=sim.graph)
            safe_plan, _ = validator.validate(plan, sim.state)
            sim.apply_signal_plan(safe_plan.assignments)

        events.tick()
        sim.step(n_ticks=1)
        sim.record_metrics(
            controller=controller,
            qaoa_energy=None,
        )
        ticks += 1

    if verbose:
        snaps = sim.metrics_history
        last = snaps[-1] if snaps else None
        if last:
            print(
                f"  [{controller:8s}] profile={profile:13s} seed={seed} "
                f"duration={duration_s}s | "
                f"mean_queue={last.mean_queue:.2f} throughput={last.throughput_total}"
            )

    df = snapshots_to_dataframe(sim.metrics_history)
    df["seed"] = seed
    df["incident"] = incident
    df["n_intersections"] = n_intersections
    return df.to_dict("records")


def main():
    parser = argparse.ArgumentParser(description="Q-Signal Benchmark Runner")
    parser.add_argument("--duration", type=int, default=600, help="Simulation seconds per run")
    parser.add_argument("--seeds", type=int, default=3, help="Number of random seeds")
    parser.add_argument("--n_intersections", type=int, default=4)
    parser.add_argument("--output", type=str, default="data/experiments/benchmark_results.csv")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seeds = list(range(args.seeds))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    all_rows = []
    total = len(CONTROLLERS) * len(DEMAND_PROFILES) * len(seeds) * len(INCIDENTS)
    done = 0
    t_start = time.time()

    for ctrl, profile, seed, incident in product(CONTROLLERS, DEMAND_PROFILES, seeds, INCIDENTS):
        print(f"[{done+1}/{total}] ctrl={ctrl} profile={profile} seed={seed} incident={incident}")
        rows = run_single(
            controller=ctrl,
            profile=profile,
            seed=seed,
            duration_s=args.duration,
            n_intersections=args.n_intersections,
            cfg=cfg,
            incident=incident,
            verbose=args.verbose,
        )
        all_rows.extend(rows)
        done += 1

    # Write CSV
    if all_rows:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)

    elapsed = time.time() - t_start
    print(f"\n✅ Benchmark complete: {len(all_rows)} rows in {elapsed:.1f}s → {args.output}")


if __name__ == "__main__":
    main()
