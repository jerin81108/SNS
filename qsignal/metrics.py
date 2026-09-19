"""
Metrics aggregation and reporting for Q-Signal.
"""

from __future__ import annotations

import statistics
from typing import Dict, List, Optional

import pandas as pd

from qsignal.models import MetricsSnapshot


def snapshots_to_dataframe(snapshots: List[MetricsSnapshot]) -> pd.DataFrame:
    """Convert a list of MetricsSnapshots to a tidy Pandas DataFrame."""
    if not snapshots:
        return pd.DataFrame()
    rows = [
        {
            "time_s": s.time_s,
            "controller": s.controller,
            "mean_queue": s.mean_queue,
            "max_queue": s.max_queue,
            "total_queue": s.total_queue,
            "throughput_total": s.throughput_total,
            "mean_wait": s.mean_wait,
            "mean_travel_time": s.mean_travel_time,
            "phase_switches": s.phase_switches,
            "estimated_fuel_L": s.estimated_fuel_L,
            "estimated_co2_kg": s.estimated_co2_kg,
            "qaoa_energy": s.qaoa_energy,
            "qaoa_gap_pct": s.qaoa_gap_pct,
            "emergency_active": s.emergency_active,
        }
        for s in snapshots
    ]
    return pd.DataFrame(rows)


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary statistics grouped by controller.
    Returns a DataFrame with mean and std of key metrics.
    """
    if df.empty:
        return pd.DataFrame()
    numeric_cols = [
        "mean_queue", "max_queue", "total_queue",
        "throughput_total", "mean_wait", "mean_travel_time",
        "phase_switches", "estimated_fuel_L", "estimated_co2_kg",
    ]
    existing = [c for c in numeric_cols if c in df.columns]
    summary = (
        df.groupby("controller")[existing]
        .agg(["mean", "std"])
        .round(3)
    )
    summary.columns = ["_".join(c) for c in summary.columns]
    return summary.reset_index()


def improvement_pct(
    baseline: float,
    optimized: float,
    lower_is_better: bool = True,
) -> float:
    """
    Compute percentage improvement of optimized vs baseline.
    For metrics where lower is better (queue, delay):
        improvement = 100 * (baseline - optimized) / baseline
    For metrics where higher is better (throughput):
        improvement = 100 * (optimized - baseline) / baseline
    """
    if baseline == 0:
        return 0.0
    if lower_is_better:
        return 100.0 * (baseline - optimized) / abs(baseline)
    else:
        return 100.0 * (optimized - baseline) / abs(baseline)


def aggregate_runs(
    all_runs: List[List[MetricsSnapshot]],
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate metric means and stds across multiple seeds/runs.
    Returns {controller: {metric: value}}.
    """
    results: Dict[str, Dict[str, List[float]]] = {}
    for run in all_runs:
        df = snapshots_to_dataframe(run)
        if df.empty:
            continue
        for controller, grp in df.groupby("controller"):
            if controller not in results:
                results[controller] = {}
            for col in ["mean_queue", "mean_wait", "throughput_total", "max_queue"]:
                if col in grp.columns:
                    results[controller].setdefault(col, []).append(grp[col].mean())

    aggregated: Dict[str, Dict[str, float]] = {}
    for ctrl, metrics in results.items():
        aggregated[ctrl] = {}
        for metric, vals in metrics.items():
            aggregated[ctrl][f"{metric}_mean"] = statistics.mean(vals) if vals else 0
            aggregated[ctrl][f"{metric}_std"] = (
                statistics.stdev(vals) if len(vals) > 1 else 0.0
            )
    return aggregated
