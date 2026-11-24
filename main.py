import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, List

from workload import generate_services, generate_dynamic_workload
from baselines import run_baseline_simulation
from adaptive_caching import run_adaptive_simulation


def compute_ci(values: List[float], confidence: float = 0.95) -> tuple[float, float]:
    """Computes mean and half-width of the 95% confidence interval."""
    arr = np.array(values, dtype=float)
    mean = float(np.mean(arr))
    if len(arr) < 2:
        return mean, 0.0
    sem = stats.sem(arr)
    ci = float(sem * stats.t.ppf((1 + confidence) / 2.0, len(arr) - 1))
    return mean, ci


def run_all_experiments(
    num_seeds: int = 15,
    num_services: int = 100,
    num_slots: int = 50,
    tasks_per_slot: int = 40,
    slot_duration: float = 2.0,
    num_edges: int = 10,
    num_clouds: int = 3
):
    os.makedirs("results", exist_ok=True)
    scenarios = ["static", "gradual", "sudden_spike"]
    methods = ["cloud_only", "static", "lru", "lfu", "greedy", "proposed"]
    method_labels = {
        "cloud_only": "Cloud-Only",
        "static": "Static (Base)",
        "lru": "LRU",
        "lfu": "LFU",
        "greedy": "Greedy",
        "proposed": "Proposed (Cost-Aware)"
    }

    total_sim_time = num_slots * slot_duration
    raw_records = []
    # Store per-slot hit ratio curves for 'sudden_spike' to plot adaptation dynamics
    spike_timelines: Dict[str, List[List[float]]] = {m: [] for m in methods}

    print(f"Running {num_seeds} random seeds across {len(scenarios)} workload scenarios...")

    for scenario in scenarios:
        print(f"  -> Evaluating scenario: {scenario.upper()}...")
        for seed_idx in range(num_seeds):
            seed = 100 + seed_idx
            services = generate_services(num_services=num_services, seed=seed)
            workload = generate_dynamic_workload(
                num_services=num_services,
                num_slots=num_slots,
                tasks_per_slot=tasks_per_slot,
                slot_duration=slot_duration,
                scenario=scenario,
                seed=seed
            )

            for m in methods:
                if m == "proposed":
                    sim_metrics = run_adaptive_simulation(
                        services=services,
                        workload_by_slot=workload,
                        num_edges=num_edges,
                        num_clouds=num_clouds,
                        slot_duration=slot_duration
                    )
                else:
                    sim_metrics = run_baseline_simulation(
                        policy=m,
                        services=services,
                        workload_by_slot=workload,
                        num_edges=num_edges,
                        num_clouds=num_clouds,
                        slot_duration=slot_duration
                    )

                summary = sim_metrics.summary(
                    num_edges=num_edges,
                    edge_cpu=20.0,
                    num_clouds=num_clouds,
                    cloud_cpu=5.0,
                    total_sim_time=total_sim_time
                )
                summary["Scenario"] = scenario
                summary["Method"] = method_labels[m]
                summary["MethodKey"] = m
                summary["Seed"] = seed
                raw_records.append(summary)

                if scenario == "sudden_spike":
                    spike_timelines[m].append(sim_metrics.slot_hit_ratios)

    df_raw = pd.DataFrame(raw_records)

    # 1. Build Summary Table with Mean ± 95% Confidence Intervals
    metric_cols = [
        "Accepted Task Ratio (%)",
        "Avg Completion Time (s)",
        "Resource Utilization (%)",
        "Cache Hit Ratio (%)",
        "Replacement Bandwidth Cost",
        "Replacement Downtime (s)",
        "Total Swaps",
        "Adaptation Time (slots)"
    ]

    summary_rows = []
    for scenario in scenarios:
        for m in methods:
            label = method_labels[m]
            sub = df_raw[(df_raw["Scenario"] == scenario) & (df_raw["MethodKey"] == m)]
            row = {"Scenario": scenario, "Method": label}
            for col in metric_cols:
                mean_val, ci_val = compute_ci(sub[col].tolist())
                row[col] = f"{mean_val:.2f} ± {ci_val:.2f}"
                row[f"{col}_mean"] = mean_val
                row[f"{col}_ci"] = ci_val
            summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    df_summary[["Scenario", "Method"] + metric_cols].to_csv("results/results_summary.csv", index=False)

    # 2. Run Paired T-Tests (Proposed vs each baseline on sudden_spike & gradual)
    stat_rows = []
    for scenario in ["gradual", "sudden_spike"]:
        prop_sub = df_raw[(df_raw["Scenario"] == scenario) & (df_raw["MethodKey"] == "proposed")].sort_values("Seed")
        for base_m in ["static", "lru", "lfu", "greedy"]:
            base_sub = df_raw[(df_raw["Scenario"] == scenario) & (df_raw["MethodKey"] == base_m)].sort_values("Seed")
            for col in ["Cache Hit Ratio (%)", "Accepted Task Ratio (%)", "Replacement Bandwidth Cost", "Avg Completion Time (s)"]:
                t_stat, p_val = stats.ttest_rel(prop_sub[col].values, base_sub[col].values)
                stat_rows.append({
                    "Scenario": scenario,
                    "Comparison": f"Proposed vs {method_labels[base_m]}",
                    "Metric": col,
                    "Proposed Mean": round(float(prop_sub[col].mean()), 3),
                    "Baseline Mean": round(float(base_sub[col].mean()), 3),
                    "t-statistic": round(float(t_stat), 3),
                    "p-value": f"{p_val:.4e}",
                    "Significant (p < 0.05)": "Yes" if p_val < 0.05 else "No"
                })

    df_stats = pd.DataFrame(stat_rows)
    df_stats.to_csv("results/statistical_tests.csv", index=False)

    print("\n=== SUMMARY RESULTS (SUDDEN SPIKE SCENARIO) ===")
    print(df_summary[df_summary["Scenario"] == "sudden_spike"][["Method"] + metric_cols].to_string(index=False))

    print("\n=== STATISTICAL SIGNIFICANCE TESTS (SUDDEN SPIKE) ===")
    print(df_stats[df_stats["Scenario"] == "sudden_spike"].to_string(index=False))

    # 3. Generate Publication-Ready Charts
    generate_plots(df_summary, spike_timelines, methods, method_labels, scenarios)
    print("\nAll CSV tables and PNG figures saved inside the 'results/' folder!")


def generate_plots(df_summary, spike_timelines, methods, method_labels, scenarios):
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = ["#7f8c8d", "#34495e", "#e67e22", "#9b59b6", "#e74c3c", "#2ecc71"]
    x = np.arange(len(scenarios))
    width = 0.13

    # Figure 1: Cache Hit Ratio & Accepted Task Ratio across Scenarios
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for idx, m in enumerate(methods):
        label = method_labels[m]
        sub = df_summary[df_summary["Method"] == label]
        
        hit_means = [sub[sub["Scenario"] == s]["Cache Hit Ratio (%)_mean"].values[0] for s in scenarios]
        hit_cis = [sub[sub["Scenario"] == s]["Cache Hit Ratio (%)_ci"].values[0] for s in scenarios]
        axes[0].bar(x + (idx - 2.5) * width, hit_means, width, yerr=hit_cis, capsize=3, label=label, color=colors[idx])

        acc_means = [sub[sub["Scenario"] == s]["Accepted Task Ratio (%)_mean"].values[0] for s in scenarios]
        acc_cis = [sub[sub["Scenario"] == s]["Accepted Task Ratio (%)_ci"].values[0] for s in scenarios]
        axes[1].bar(x + (idx - 2.5) * width, acc_means, width, yerr=acc_cis, capsize=3, label=label, color=colors[idx])

    axes[0].set_title("Cache Hit Ratio Across Workloads (95% CI)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Cache Hit Ratio (%)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["Static", "Gradual Drift", "Sudden Spike"])
    axes[0].legend(fontsize=8)

    axes[1].set_title("Accepted Task Ratio / Deadline Satisfaction (95% CI)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Accepted Tasks (%)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["Static", "Gradual Drift", "Sudden Spike"])
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("results/fig1_hit_and_acceptance.png", dpi=300)
    plt.close()

    # Figure 2: Replacement Overhead (Bandwidth Cost & Downtime) on Dynamic Workloads
    dyn_scenarios = ["gradual", "sudden_spike"]
    active_methods = ["static", "lru", "lfu", "greedy", "proposed"]
    x_dyn = np.arange(len(dyn_scenarios))
    w_dyn = 0.15

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for idx, m in enumerate(active_methods):
        label = method_labels[m]
        sub = df_summary[df_summary["Method"] == label]
        bw_means = [sub[sub["Scenario"] == s]["Replacement Bandwidth Cost_mean"].values[0] for s in dyn_scenarios]
        bw_cis = [sub[sub["Scenario"] == s]["Replacement Bandwidth Cost_ci"].values[0] for s in dyn_scenarios]
        axes[0].bar(x_dyn + (idx - 2) * w_dyn, bw_means, w_dyn, yerr=bw_cis, capsize=3, label=label, color=colors[idx + 1])

        dt_means = [sub[sub["Scenario"] == s]["Replacement Downtime (s)_mean"].values[0] for s in dyn_scenarios]
        dt_cis = [sub[sub["Scenario"] == s]["Replacement Downtime (s)_ci"].values[0] for s in dyn_scenarios]
        axes[1].bar(x_dyn + (idx - 2) * w_dyn, dt_means, w_dyn, yerr=dt_cis, capsize=3, label=label, color=colors[idx + 1])

    axes[0].set_title("Replacement Bandwidth Overhead (Lower is Better)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Total Bandwidth Cost Units")
    axes[0].set_xticks(x_dyn)
    axes[0].set_xticklabels(["Gradual Drift", "Sudden Spike"])
    axes[0].legend(fontsize=8)

    axes[1].set_title("Cumulative Service Retrieval Downtime (Lower is Better)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Retrieval Delay (seconds)")
    axes[1].set_xticks(x_dyn)
    axes[1].set_xticklabels(["Gradual Drift", "Sudden Spike"])
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("results/fig2_replacement_overhead.png", dpi=300)
    plt.close()

    # Figure 3: Adaptation Timeline During Sudden Popularity Spike
    plt.figure(figsize=(11, 5))
    plot_methods = ["static", "lru", "lfu", "greedy", "proposed"]
    line_styles = ["--", ":", "-.", "--", "-"]
    line_widths = [2.0, 1.8, 1.8, 1.8, 2.8]

    for idx, m in enumerate(plot_methods):
        curves = np.array(spike_timelines[m])  # shape: (num_seeds, num_slots)
        mean_curve = np.mean(curves, axis=0)
        # Smooth slightly with a 2-slot rolling average for visual clarity
        smoothed = np.convolve(mean_curve, np.ones(2)/2, mode="same")
        plt.plot(
            range(len(smoothed)),
            smoothed,
            label=method_labels[m],
            color=colors[idx + 1],
            linestyle=line_styles[idx],
            linewidth=line_widths[idx]
        )

    plt.axvline(x=25, color="black", linestyle="--", alpha=0.7, label="Popularity Shift #1 (Slot 25)")
    plt.axvline(x=37, color="gray", linestyle=":", alpha=0.7, label="Popularity Shift #2 (Slot 37)")
    plt.title("Cache Hit Ratio Adaptation Under Sudden Demand Shifts", fontsize=13, fontweight="bold")
    plt.xlabel("Simulation Time Slot")
    plt.ylabel("Cache Hit Ratio (%)")
    plt.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    plt.savefig("results/fig3_adaptation_timeline.png", dpi=300)
    plt.close()

    # Figure 4: Task Completion Time & Resource Utilization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for idx, m in enumerate(methods):
        label = method_labels[m]
        sub = df_summary[df_summary["Method"] == label]
        ct_means = [sub[sub["Scenario"] == s]["Avg Completion Time (s)_mean"].values[0] for s in scenarios]
        ct_cis = [sub[sub["Scenario"] == s]["Avg Completion Time (s)_ci"].values[0] for s in scenarios]
        axes[0].bar(x + (idx - 2.5) * width, ct_means, width, yerr=ct_cis, capsize=3, label=label, color=colors[idx])

        ut_means = [sub[sub["Scenario"] == s]["Resource Utilization (%)_mean"].values[0] for s in scenarios]
        ut_cis = [sub[sub["Scenario"] == s]["Resource Utilization (%)_ci"].values[0] for s in scenarios]
        axes[1].bar(x + (idx - 2.5) * width, ut_means, width, yerr=ut_cis, capsize=3, label=label, color=colors[idx])

    axes[0].set_title("Average Task Completion Time (Lower is Better)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Completion Time (seconds)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["Static", "Gradual Drift", "Sudden Spike"])
    axes[0].legend(fontsize=8)

    axes[1].set_title("Overall Compute Resource Utilization", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Resource Utilization (%)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["Static", "Gradual Drift", "Sudden Spike"])
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("results/fig4_completion_and_utilization.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    run_all_experiments()