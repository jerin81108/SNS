"""
Benchmark results analyser for Q-Signal.
Loads CSV results, computes improvement statistics, and generates summary tables.

Usage:
    python experiments/analyze_results.py --input data/experiments/benchmark_results.csv
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qsignal.metrics import improvement_pct


def load_results(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def summary_by_controller(df: pd.DataFrame) -> pd.DataFrame:
    """Mean and std of key metrics, grouped by controller."""
    cols = ["mean_queue", "max_queue", "mean_wait", "throughput_total",
            "estimated_co2_kg", "phase_switches"]
    existing = [c for c in cols if c in df.columns]
    agg = df.groupby(["controller", "incident"])[existing].agg(["mean", "std"]).round(3)
    agg.columns = ["_".join(c) for c in agg.columns]
    return agg.reset_index()


def improvement_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Compute percentage improvements of QAOA and pressure vs fixed-time baseline."""
    rows = []
    for incident, grp in summary.groupby("incident"):
        grp = grp.set_index("controller")
        if "fixed" not in grp.index:
            continue
        for ctrl in ["pressure", "qaoa"]:
            if ctrl not in grp.index:
                continue
            row = {"incident": incident, "controller": ctrl}
            for metric, lower_better in [
                ("mean_queue_mean", True),
                ("mean_wait_mean", True),
                ("throughput_total_mean", False),
                ("estimated_co2_kg_mean", True),
            ]:
                if metric in grp.columns:
                    baseline = grp.loc["fixed", metric]
                    optimized = grp.loc[ctrl, metric]
                    row[f"{metric}_improvement_%"] = round(
                        improvement_pct(baseline, optimized, lower_better), 2
                    )
            rows.append(row)
    return pd.DataFrame(rows)


def plot_improvement(improvement_df: pd.DataFrame, output_path: str) -> None:
    """Bar chart of improvement percentages."""
    if improvement_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0f172a")
    ax.set_facecolor("#1e293b")
    ax.tick_params(colors="#94a3b8")
    ax.spines[:].set_color("#334155")

    metric_cols = [c for c in improvement_df.columns if c.endswith("_improvement_%")]
    x = range(len(improvement_df))
    width = 0.18
    palette = ["#22c55e", "#f59e0b", "#3b82f6", "#a855f7"]

    for i, metric in enumerate(metric_cols[:4]):
        offsets = [xi + (i - 1.5) * width for xi in x]
        vals = improvement_df[metric].tolist()
        ax.bar(offsets, vals, width=width, label=metric.replace("_improvement_%", "").replace("_", " "),
               color=palette[i], alpha=0.85)

    ax.axhline(0, color="#64748b", linewidth=0.8, linestyle="--")
    labels = [f"{r['controller']} / {r['incident']}" for _, r in improvement_df.iterrows()]
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right", color="#94a3b8")
    ax.set_ylabel("Improvement over Fixed (%)", color="#94a3b8")
    ax.set_title("Controller Improvement vs Fixed-Time Baseline", color="#f1f5f9", fontsize=12)
    ax.legend(facecolor="#1e293b", labelcolor="#f1f5f9", fontsize=8)
    ax.grid(True, axis="y", color="#334155", linewidth=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, facecolor="#0f172a")
    print(f"  Chart saved: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Q-Signal Results Analyser")
    parser.add_argument("--input", type=str, default="data/experiments/benchmark_results.csv")
    parser.add_argument("--output_dir", type=str, default="data/experiments/")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ Results file not found: {args.input}")
        print("   Run experiments/run_benchmark.py first.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    df = load_results(args.input)
    print(f"✅ Loaded {len(df)} rows from {args.input}")

    # Summary
    summary = summary_by_controller(df)
    summary_path = os.path.join(args.output_dir, "summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\n📊 Summary by controller:\n{summary.to_string()}")
    print(f"\n  Summary saved: {summary_path}")

    # Improvement table
    imp = improvement_table(summary)
    if not imp.empty:
        imp_path = os.path.join(args.output_dir, "improvements.csv")
        imp.to_csv(imp_path, index=False)
        print(f"\n📈 Improvement over fixed-time:\n{imp.to_string()}")
        print(f"\n  Improvements saved: {imp_path}")

    # Plot
    chart_path = os.path.join(args.output_dir, "improvement_chart.png")
    plot_improvement(imp, chart_path)


if __name__ == "__main__":
    main()
