# Cost-Aware Adaptive Service Caching and Task Offloading in Dynamic Edge-Cloud Computing Environments

A discrete-time simulation framework and interactive browser visualization for joint **service caching** and **deadline-aware task offloading** under non-stationary (time-varying) workloads. This project extends the static edge-cloud model of *Gao et al. (IEEE Access, 2022 — LMPSO)* to handle gradual popularity drift and sudden demand spikes while accounting for **service replacement overhead** (backhaul bandwidth consumption and retrieval downtime).

---

## Key Features

* **Dynamic Edge-Cloud System Model (`models.py`):** Models heterogeneous Edge Servers ($20\text{ GHz}$ CPU, $300\text{ GB}$ storage, $10$ channels at $60\text{ Mbps}$) and remote Cloud Servers ($5\text{ GHz}$ CPU, $\infty$ storage, $15\text{ Mbps}$), accounting for both network transmission and compute queueing bottlenecks as well as service download readiness delays.
* **Non-Stationary Zipf Workload Generator (`workload.py`):** Synthesizes time-slotted IoT/mobile request streams across three distinct traffic regimes:
  * `static`: Stationary Zipf popularity distribution (baseline validation).
  * `gradual`: Continuous rank drift simulating evolving user trends.
  * `sudden_spike`: Abrupt popularity inversions midway through execution simulating flash crowds or emergency events.
* **Cost-Aware Adaptive Caching Controller (`adaptive_caching.py`):**
  * Tracks service demand using an **Exponentially Weighted Moving Average (EWMA)** estimator ($\gamma$).
  * Evaluates candidate replacements via a **Net Benefit Density vs. Replacement Penalty Density** utility rule, amortizing backhaul bandwidth and retrieval downtime over a lookahead window ($H$) to prevent cache thrashing during transient noise.
* **Comprehensive Baseline Suite (`baselines.py` & `lmpso.py`):** Benchmarks against **Cloud-Only**, **Static Caching**, **LRU**, **LFU**, **Popularity-Based Greedy**, and **LMPSO** (Particle Swarm Optimization with Lévy Flight).
* **Statistical Evaluation & Sensitivity Analysis (`main.py` & `sensitivity.py`):** Runs multi-seed experiments ($15$ seeds) with $95\%$ confidence intervals, paired $t$-tests ($p < 0.05$), parameter sweeps ($\lambda$, Zipf $\alpha$), and high-resolution Matplotlib exports.
* **Zero-Dependency Browser Demo (`demo.html`):** Interactive HTML5/Canvas dashboard for live visualization of task routing, edge storage utilization, and real-time cache adaptation.

---

## Project Structure

```text
edge_cloud_project/
├── models.py              # EdgeServer, CloudServer, Service, and Task definitions
├── workload.py            # Static, Gradual Drift, and Sudden Spike Zipf workload generator
├── baselines.py           # Offloading engine, 5 baseline policies, and 7 evaluation metrics
├── lmpso.py               # Base paper LMPSO meta-heuristic implementation + runtime tracking
├── adaptive_caching.py    # Proposed Cost-Aware Adaptive Caching & Task Offloading framework
├── main.py                # Multi-seed benchmarking, 95% CIs, paired t-tests, and Figures 1–4
├── sensitivity.py         # LMPSO runtime comparison & sensitivity sweeps (Figures 5–6)
├── demo.html              # Self-contained interactive browser simulator & results viewer
├── results/               # Generated CSV tables and publication-ready PNG plots
│   ├── results_summary.csv
│   ├── statistical_tests.csv
│   ├── lmpso_comparison.csv
│   └── fig1_... to fig6_...png
└── README.md
```

---

## Getting Started

### 1. Prerequisites
* Python 3.10 or higher
* A modern web browser (Chrome, Edge, Firefox) for `demo.html`

### 2. Installation (VS Code / Terminal)

```bash
# Navigate to the project folder
cd edge_cloud_project

# Create and activate a virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate
# macOS / Linux:
# source venv/bin/activate

# Install required scientific & plotting packages
pip install numpy pandas matplotlib scipy
```

---

## Running the Simulations

### 1. Full Multi-Seed Benchmark & Statistical Tests
Runs all 6 strategies across 15 random seeds and 3 workload scenarios, exporting `results_summary.csv`, `statistical_tests.csv`, and Figures 1–4 into `results/`:
```bash
python main.py
```

### 2. LMPSO Runtime Comparison & Sensitivity Sweeps
Compares update latency (ms) against Gao et al.'s LMPSO and sweeps replacement cost weight ($\lambda$) and Zipf skewness ($\alpha$), generating `lmpso_comparison.csv` and Figures 5–6:
```bash
python sensitivity.py
```

### 3. Interactive Browser Demo
Double-click `demo.html` (or open it in any browser) to run the live interactive simulation, adjust $\lambda$, $\gamma$, and edge storage capacity in real time, trigger manual demand spikes, and inspect the 15-seed benchmark table.

---

## Summary of Experimental Results (Sudden Spike Scenario)

Evaluated across **15 random seeds** ($100$ services, $10$ Edge Servers, $3$ Cloud Servers, $50$ time slots, $2,000$ tasks):

| Caching Strategy | Cache Hit Ratio (%) | Accepted Tasks (%) | Avg Completion (s) | Replacement BW Cost | Decision Time / Update |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Static Caching (Base Paper)** | $27.13\%$ | $32.42\%$ | $1.370\text{ s}$ | $0$ | $441.46\text{ ms}$ (LMPSO) |
| **Greedy (Cost-Blind)** | $44.11\%$ | $47.91\%$ | $1.290\text{ s}$ | $85,062$ | $0.05\text{ ms}$ |
| **LRU (Reactive)** | $72.85\%$ | $76.51\%$ | $1.081\text{ s}$ | $32,807$ | $0.08\text{ ms}$ |
| **LFU (Reactive)** | $57.47\%$ | $61.85\%$ | $1.138\text{ s}$ | $51,607$ | $0.08\text{ ms}$ |
| **Proposed (Cost-Aware Adaptive)** | **$54.16\%$** | **$58.89\%$** | **$1.134\text{ s}$** | **$12,840$** | **$1.38\text{ ms}$** |

### Key Findings
1. **Vs. Static Caching:** Nearly doubles the cache hit ratio ($+99.6\%$ relative gain, $p = 4.88 \times 10^{-18}$) and increases deadline satisfaction by $+26.47$ percentage points under sudden demand shifts.
2. **Vs. Cost-Blind Greedy:** Outperforms Greedy on hit ratio ($+10.05\%$) and task acceptance ($+10.98\%$) while cutting replacement bandwidth overhead by **$84.9\%$** ($p = 1.79 \times 10^{-19}$).
3. **Vs. Reactive Policies (LRU/LFU):** Eliminates severe cache thrashing, reducing backhaul bandwidth cost by **$59\%$ to $73\%$** while maintaining the lowest average task completion time ($1.134\text{ s}$).
4. **Vs. Meta-Heuristic LMPSO:** Executes cache adaptation decisions in **$1.38\text{ ms}$** (~$300\times$ faster than LMPSO's $402\text{--}441\text{ ms}$), enabling real-time online edge orchestration.

---

## Reference
* T. Gao, Q. Tang, J. Li, Y. Zhang, Y. Li, and J. Zhang, *"A Particle Swarm Optimization With Lévy Flight for Service Caching and Task Offloading in Edge-Cloud Computing,"* IEEE Access, vol. 10, pp. 76636–76647, 2022.