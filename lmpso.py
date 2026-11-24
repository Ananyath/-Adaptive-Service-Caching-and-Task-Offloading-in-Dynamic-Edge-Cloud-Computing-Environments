import math
import time
import numpy as np
from typing import Dict, List
from models import Service, Task, EdgeServer, CloudServer
from baselines import SimulationMetrics, offload_task, install_service_on_edge
import baselines


def _levy_step(beta: float = 1.5, size: int = 1, rng: np.random.Generator = None) -> np.ndarray:
    """Implements Lévy flight step size from Gao et al. (2022) Eq. (24)."""
    if rng is None:
        rng = np.random.default_rng()
    num = math.gamma(1.0 + beta) * math.sin(math.pi * beta / 2.0)
    den = beta * math.gamma((1.0 + beta) / 2.0) * (2.0 ** ((beta - 1.0) / 2.0))
    sigma_u = (num / den) ** (1.0 / beta)
    u = rng.normal(0.0, sigma_u, size=size)
    v = rng.normal(0.0, 1.0, size=size)
    return u / (np.abs(v) ** (1.0 / beta))


def optimize_cache_lmpso(
    services: Dict[int, Service],
    recent_tasks: List[Task],
    num_edges: int = 10,
    num_clouds: int = 3,
    num_particles: int = 25,
    max_iters: int = 15,
    seed: int = 42
) -> List[set]:
    """
    Implements LMPSO (Gao et al., IEEE Access 2022) to optimize service caching placement
    on Edge Servers given a workload batch, maximizing Objective Eq. (17): N + U.
    Returns a list of cached service ID sets, one per Edge Server.
    """
    rng = np.random.default_rng(seed)
    num_services = len(services)
    
    # Filter candidate services to those actually appearing in recent workload for fast convergence
    active_sids = list({t.service_id for t in recent_tasks})
    if not active_sids:
        active_sids = list(services.keys())[:20]

    # Particle position matrix: shape (num_particles, num_edges, len(active_sids)) in [0, 1]
    # Higher continuous value = higher priority to cache that service on that edge
    positions = rng.uniform(0.0, 1.0, size=(num_particles, num_edges, len(active_sids)))
    velocities = rng.uniform(-0.2, 0.2, size=positions.shape)

    pbest_pos = positions.copy()
    pbest_fit = np.full(num_particles, -1.0)
    gbest_pos = positions[0].copy()
    gbest_fit = -1.0

    def decode_and_evaluate(pos_matrix: np.ndarray) -> tuple[float, List[set]]:
        caches: List[set] = []
        test_edges = [EdgeServer(server_id=i) for i in range(num_edges)]
        test_clouds = [CloudServer(server_id=i) for i in range(num_clouds)]

        for e_idx in range(num_edges):
            order = np.argsort(-pos_matrix[e_idx])
            for idx in order:
                sid = active_sids[idx]
                if test_edges[e_idx].free_storage(services) >= services[sid].storage_size:
                    test_edges[e_idx].cached_services.add(sid)
            caches.append(set(test_edges[e_idx].cached_services))

        # Evaluate fitness Eq. (17): N + U on the sample tasks
        dummy_metrics = SimulationMetrics(num_slots=1)
        for task in recent_tasks:
            offload_task(task, test_edges, test_clouds, dummy_metrics)

        n_accepted = dummy_metrics.accepted_tasks
        total_cap = (num_edges * 20.0 + num_clouds * 5.0) * 2.0
        u_util = min(0.99, dummy_metrics.total_used_cpu_cycles / max(1.0, total_cap))
        return float(n_accepted) + u_util, caches

    best_caches: List[set] = [set() for _ in range(num_edges)]

    for it in range(max_iters):
        # Linearly decreasing inertia weight from 0.9 to 0.4 (Gao et al. Sec IV-A)
        w = 0.9 - (0.5 * it / max(1, max_iters - 1))
        c1, c2 = 2.0, 2.0

        for p in range(num_particles):
            fit, caches = decode_and_evaluate(positions[p])
            if fit > pbest_fit[p]:
                pbest_fit[p] = fit
                pbest_pos[p] = positions[p].copy()
            if fit > gbest_fit:
                gbest_fit = fit
                gbest_pos = positions[p].copy()
                best_caches = caches

        # PSO velocity & position update + Lévy Flight jump (Eq. 21-23)
        for p in range(num_particles):
            r1 = rng.uniform(0.0, 1.0, size=positions[p].shape)
            r2 = rng.uniform(0.0, 1.0, size=positions[p].shape)
            velocities[p] = (
                w * velocities[p]
                + c1 * r1 * (pbest_pos[p] - positions[p])
                + c2 * r2 * (gbest_pos - positions[p])
            )
            positions[p] = np.clip(positions[p] + velocities[p], 0.0, 1.0)

            # Lévy flight perturbation with probability R = 0.3
            if rng.uniform() < 0.3:
                levy = _levy_step(beta=1.5, size=1, rng=rng)[0] * 0.05
                positions[p] = np.clip(
                    positions[p] + levy * (gbest_pos - positions[p]),
                    0.0,
                    1.0
                )

    return best_caches


def run_lmpso_simulation(
    services: Dict[int, Service],
    workload_by_slot: List[List[Task]],
    num_edges: int = 10,
    num_clouds: int = 3,
    slot_duration: float = 2.0,
    dynamic_reoptimize: bool = True,
    seed: int = 42
) -> tuple[SimulationMetrics, float]:
    """
    Runs LMPSO from Gao et al. (2022).
    - If dynamic_reoptimize=False: Static LMPSO (solves once at slot 0, as in the base paper).
    - If dynamic_reoptimize=True: Re-runs LMPSO every 5 slots to adapt to dynamic workloads.
    Returns (SimulationMetrics, average_decision_time_ms).
    """
    baselines._SERVICE_REGISTRY = services
    edges = [EdgeServer(server_id=i) for i in range(num_edges)]
    clouds = [CloudServer(server_id=i) for i in range(num_clouds)]
    metrics = SimulationMetrics(num_slots=len(workload_by_slot))
    decision_times_ms: List[float] = []

    # Initial LMPSO placement on slot 0
    t0 = time.perf_counter()
    init_caches = optimize_cache_lmpso(services, workload_by_slot[0], num_edges, num_clouds, seed=seed)
    decision_times_ms.append((time.perf_counter() - t0) * 1000.0)

    for e_idx, es in enumerate(edges):
        for sid in init_caches[e_idx]:
            es.cached_services.add(sid)
            es.service_ready_time[sid] = 0.0

    for slot_idx, slot_tasks in enumerate(workload_by_slot):
        slot_hits_before = metrics.cache_hits
        slot_misses_before = metrics.cache_misses
        slot_acc_before = metrics.accepted_tasks

        for task in slot_tasks:
            offload_task(task, edges, clouds, metrics)

        # Dynamic LMPSO re-runs meta-heuristic optimization periodically (every 5 slots)
        if dynamic_reoptimize and slot_idx > 0 and slot_idx % 5 == 0 and slot_idx < len(workload_by_slot) - 1:
            slot_end = (slot_idx + 1) * slot_duration
            t_start = time.perf_counter()
            target_caches = optimize_cache_lmpso(
                services, slot_tasks, num_edges, num_clouds, seed=seed + slot_idx
            )
            decision_times_ms.append((time.perf_counter() - t_start) * 1000.0)

            for e_idx, es in enumerate(edges):
                desired = target_caches[e_idx]
                victims = [s for s in es.cached_services if s not in desired]
                for sid in desired:
                    if sid not in es.cached_services:
                        install_service_on_edge(es, services[sid], slot_end, metrics, victims)

        s_hits = metrics.cache_hits - slot_hits_before
        s_misses = metrics.cache_misses - slot_misses_before
        s_total = s_hits + s_misses
        metrics.slot_hit_ratios.append((s_hits / s_total * 100.0) if s_total > 0 else 0.0)
        metrics.slot_accept_ratios.append(
            ((metrics.accepted_tasks - slot_acc_before) / len(slot_tasks) * 100.0) if slot_tasks else 0.0
        )

    return metrics, float(np.mean(decision_times_ms))