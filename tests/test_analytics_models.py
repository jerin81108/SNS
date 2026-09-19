"""
Tests for AI Analytics and Carbon Simulation Models (co2.csv and traffic.csv).
"""

import os
import pytest
import pandas as pd
from qsignal.analytics_models import CO2EmissionModel, TrafficSimulationModel, SimulationResult


def test_co2_emission_model_training():
    model = CO2EmissionModel()
    assert model.is_trained is True
    assert len(model.category_stats) > 0

    # Verify standard categories exist
    assert "Sedan / Compact" in model.category_stats
    assert "SUV" in model.category_stats

    # Check that idle CO2 rate is positive and reasonable (0.4 to 2.5 g/s)
    fleet_idle_rate = model.get_fleet_average_idle_g_s()
    assert 0.4 <= fleet_idle_rate <= 2.5

    # Verify high predictive accuracy on vehicle emissions & fuel consumption (> 0.90)
    assert model.r2_co2 > 0.90
    assert model.r2_fuel_city > 0.90


def test_traffic_simulation_model_training():
    model = TrafficSimulationModel()
    assert model.is_trained is True
    # Verify high predictive accuracy on traffic flow (> 0.90)
    assert model.r2_score > 0.90


def test_simulate_month_emissions():
    co2_model = CO2EmissionModel()
    traffic_model = TrafficSimulationModel()

    res: SimulationResult = traffic_model.simulate_month(
        co2_model=co2_model,
        start_month=10,
        junction=1,
        base_wait_seconds=45.0,
        qaoa_wait_reduction=0.35,
        random_seed=42,
    )

    # 1. Verify daily_df has exactly 30 days
    assert len(res.daily_df) == 30
    assert "total_vehicles" in res.daily_df.columns
    assert "baseline_co2_kg" in res.daily_df.columns
    assert "optimized_co2_kg" in res.daily_df.columns
    assert "co2_saved_kg" in res.daily_df.columns

    # 2. Verify all 30 days have non-negative traffic and emissions
    assert (res.daily_df["total_vehicles"] > 0).all()
    assert (res.daily_df["baseline_co2_kg"] > 0).all()
    assert (res.daily_df["optimized_co2_kg"] > 0).all()
    assert (res.daily_df["co2_saved_kg"] > 0).all()

    # 3. Verify hourly_df has 24 * 30 = 720 rows
    assert len(res.hourly_df) == 720

    # 4. Verify optimization savings consistency
    assert (res.daily_df["baseline_co2_kg"] > res.daily_df["optimized_co2_kg"]).all()

    # 5. Verify stats dictionary contents
    stats = res.stats
    assert stats["total_month_vehicles"] > 0
    assert stats["total_baseline_co2_kg"] > stats["total_optimized_co2_kg"]
    assert stats["reduction_pct"] > 25.0
    assert stats["trees_saved_month"] > 0
    assert stats["r2_score"] > 0.90
    assert stats["co2_r2_score"] > 0.90

