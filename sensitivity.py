import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from workload import generate_services, generate_dynamic_workload
from baselines import run_baseline_simulation
from adaptive_caching import run_adaptive_simulation
from lmpso import run_lmpso_simulation


def run_sensitivity_experiments():
    os.makedirs("results", exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    # ---------------------------------------------------------
    # 1. Compare Decision Runtime (ms) & Performance vs LMPSO
    # ---------------------------------------------------------
    print("1. Benchmarking Proposed Framework vs Base Paper LMPSO...")
    services = generate_services(num_services=100, seed=42)
    workload = generate_dynamic_workload(scenario="sudden_spike", seed=42)

    m_lmpso_static, rt_lmpso_s = run_lmpso_simulation(services, workload, dynamic_reoptimize=False)
    m_lmpso_dyn, rt_lmpso_d = run_lmpso_simulation(services, workload, dynamic_reoptimize=True)
    m_greedy = run_baseline_simulation("greedy", services, workload)

    t0 = time.perf_counter()
    m_proposed = run_adaptive_simulation(services, workload)
    rt_proposed_ms = ((time.perf_counter() - t0) * 1000.0) / len(workload)

    comp_rows = [
        {"Method": "LMPSO (Static - Base Paper)", **m_lmpso_static.summary(10, 20.0, 3, 5.0, 100.0), "Decision Time/Update (ms)": round(rt_lmpso_s, 2)},
        {"Method": "LMPSO (Dynamic Re-Opt)", **m_lmpso_dyn.summary(10, 20.0, 3, 5.0, 100.0), "Decision Time/Update (ms)": round(rt_lmpso_d, 2)},
        {"Method": "Greedy (Cost-Blind)", **m_greedy.summary(10, 20.0, 3, 5.0, 100.0), "Decision Time/Update (ms)": 0.05},
        {"Method": "Proposed (Cost-Aware)", **m_proposed.summary(10, 20.0, 3, 5.0, 100.0), "Decision Time/Update (ms)": round(rt_proposed_ms, 3)},
    ]
    df_lmpso = pd.DataFrame(comp_rows)
    df_lmpso.to_csv("results/lmpso_comparison.csv", index=False)
    print(df_lmpso[["Method", "Cache Hit Ratio (%)", "Accepted Task Ratio (%)", "Replacement Bandwidth Cost", "Decision Time/Update (ms)"]].to_string(index=False))

    # ---------------------------------------------------------
    # 2. Sensitivity to Replacement Cost Weight (Pareto Curve)
    # ---------------------------------------------------------
    print("\n2. Running Replacement Cost Weight (lambda) Sensitivity Sweep...")
    weights = [0.0, 0.02, 0.05, 0.08, 0.12, 0.20, 0.35]
    hit_rates, bw_costs, accept_rates = [], [], []

    for w in weights:
        m = run_adaptive_simulation(services, workload, cost_weight=w)
        s = m.summary(10, 20.0, 3, 5.0, 100.0)
        hit_rates.append(s["Cache Hit Ratio (%)"])
        bw_costs.append(s["Replacement Bandwidth Cost"])
        accept_rates.append(s["Accepted Task Ratio (%)"])

    fig, ax1 = plt.subplots(figsize=(9, 5))
    color1 = "#2ecc71"
    ax1.set_xlabel("Replacement Cost Weight (cost_weight)", fontsize=11)
    ax1.set_ylabel("Cache Hit Ratio (%)", color=color1, fontsize=11, fontweight="bold")
    ax1.plot(weights, hit_rates, marker="o", color=color1, linewidth=2.5, label="Cache Hit Ratio (%)")
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = "#e74c3c"
    ax2.set_ylabel("Replacement Bandwidth Cost", color=color2, fontsize=11, fontweight="bold")
    ax2.plot(weights, bw_costs, marker="s", linestyle="--", color=color2, linewidth=2.5, label="Bandwidth Cost")
    ax2.tick_params(axis="y", labelcolor=color2)

    plt.title("Trade-Off: Cache Hit Ratio vs. Replacement Overhead", fontsize=12, fontweight="bold")
    fig.tight_layout()
    plt.savefig("results/fig5_cost_weight_tradeoff.png", dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # 3. Sensitivity to Zipf Workload Skewness (alpha)
    # ---------------------------------------------------------
    print("3. Running Zipf Skewness (alpha) Sensitivity Sweep...")
    alphas = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
    res_static, res_greedy, res_prop = [], [], []

    for a in alphas:
        wl = generate_dynamic_workload(scenario="sudden_spike", zipf_alpha=a, seed=42)
        res_static.append(run_baseline_simulation("static", services, wl).summary(10, 20.0, 3, 5.0, 100.0)["Cache Hit Ratio (%)"])
        res_greedy.append(run_baseline_simulation("greedy", services, wl).summary(10, 20.0, 3, 5.0, 100.0)["Cache Hit Ratio (%)"])
        res_prop.append(run_adaptive_simulation(services, wl).summary(10, 20.0, 3, 5.0, 100.0)["Cache Hit Ratio (%)"])

    plt.figure(figsize=(9, 5))
    plt.plot(alphas, res_static, marker="^", linestyle="--", label="Static (Base)", color="#34495e", linewidth=2)
    plt.plot(alphas, res_greedy, marker="s", linestyle="-.", label="Greedy", color="#e74c3c", linewidth=2)
    plt.plot(alphas, res_prop, marker="o", linestyle="-", label="Proposed (Cost-Aware)", color="#2ecc71", linewidth=2.8)
    plt.title("Cache Hit Ratio Across Workload Skewness (Zipf Alpha)", fontsize=12, fontweight="bold")
    plt.xlabel("Zipf Distribution Parameter (alpha)", fontsize=11)
    plt.ylabel("Cache Hit Ratio (%)", fontsize=11)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/fig6_zipf_sensitivity.png", dpi=300)
    plt.close()

    print("\nSaved 'results/lmpso_comparison.csv', 'fig5_cost_weight_tradeoff.png', and 'fig6_zipf_sensitivity.png'!")


if __name__ == "__main__":
    run_sensitivity_experiments()