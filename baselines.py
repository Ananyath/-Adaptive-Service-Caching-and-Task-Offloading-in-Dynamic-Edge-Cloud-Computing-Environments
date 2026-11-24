import copy
import numpy as np
from typing import Dict, List, Set, Tuple
from models import Service, Task, EdgeServer, CloudServer


class SimulationMetrics:
    """Tracks all 7 performance metrics specified in the research proposal."""
    def __init__(self, num_slots: int):
        self.total_tasks = 0
        self.accepted_tasks = 0
        self.edge_accepted = 0
        self.cloud_accepted = 0
        self.completion_times: List[float] = []
        
        # Cache & Replacement metrics
        self.cache_hits = 0
        self.cache_misses = 0
        self.replacement_count = 0
        self.total_bandwidth_cost = 0.0
        self.total_retrieval_delay = 0.0
        
        # Per-slot tracking (used to measure adaptation time after workload shifts)
        self.slot_hit_ratios: List[float] = []
        self.slot_accept_ratios: List[float] = []
        
        # Resource utilization tracking
        self.total_used_cpu_cycles = 0.0
        self.num_slots = num_slots

    def summary(self, num_edges: int, edge_cpu: float, num_clouds: int, cloud_cpu: float, total_sim_time: float) -> Dict[str, float]:
        accept_ratio = (self.accepted_tasks / self.total_tasks * 100.0) if self.total_tasks > 0 else 0.0
        avg_completion = float(np.mean(self.completion_times)) if self.completion_times else 0.0
        total_lookups = self.cache_hits + self.cache_misses
        hit_ratio = (self.cache_hits / total_lookups * 100.0) if total_lookups > 0 else 0.0
        
        total_available_cpu = (num_edges * edge_cpu + num_clouds * cloud_cpu) * total_sim_time
        utilization = (self.total_used_cpu_cycles / total_available_cpu * 100.0) if total_available_cpu > 0 else 0.0

        # Adaptation time: slots needed after midpoint spike (slot 25) to reach 80% of pre-spike hit ratio
        adaptation_slots = 0
        mid = self.num_slots // 2
        if len(self.slot_hit_ratios) > mid + 2:
            pre_spike_avg = np.mean(self.slot_hit_ratios[max(0, mid - 5):mid])
            target = 0.80 * pre_spike_avg
            if target > 0:
                for offset, hr in enumerate(self.slot_hit_ratios[mid:]):
                    if hr >= target:
                        adaptation_slots = offset
                        break
                else:
                    adaptation_slots = len(self.slot_hit_ratios) - mid

        return {
            "Accepted Task Ratio (%)": round(accept_ratio, 2),
            "Deadline Satisfaction (%)": round(accept_ratio, 2),
            "Avg Completion Time (s)": round(avg_completion, 4),
            "Resource Utilization (%)": round(utilization, 2),
            "Cache Hit Ratio (%)": round(hit_ratio, 2),
            "Replacement Bandwidth Cost": round(self.total_bandwidth_cost, 1),
            "Replacement Downtime (s)": round(self.total_retrieval_delay, 2),
            "Total Swaps": self.replacement_count,
            "Adaptation Time (slots)": adaptation_slots
        }


def offload_task(
    task: Task,
    edges: List[EdgeServer],
    clouds: List[CloudServer],
    metrics: SimulationMetrics
) -> Tuple[bool, str]:
    """
    Task Offloading Engine:
    1. Check all Edge Servers that have the service cached; pick the one finishing earliest within deadline.
    2. Otherwise, check Cloud Servers; pick the one finishing earliest within deadline.
    3. Otherwise, reject the task.
    """
    metrics.total_tasks += 1

    # Check if service is cached on any edge
    candidate_edges = [es for es in edges if task.service_id in es.cached_services]
    if candidate_edges:
        metrics.cache_hits += 1
        best_es = min(candidate_edges, key=lambda es: es.estimate_finish_time(task)[0])
        est_finish, _ = best_es.estimate_finish_time(task)
        if best_es.assign_task(task):
            metrics.accepted_tasks += 1
            metrics.edge_accepted += 1
            metrics.completion_times.append(est_finish - task.arrival_time)
            metrics.total_used_cpu_cycles += task.cpu_demand
            return True, "edge"
    else:
        metrics.cache_misses += 1

    # Fallback to Cloud Server (pick cloud finishing earliest)
    best_cs = min(clouds, key=lambda cs: cs.estimate_finish_time(task))
    est_cloud_finish = best_cs.estimate_finish_time(task)
    if best_cs.assign_task(task):
        metrics.accepted_tasks += 1
        metrics.cloud_accepted += 1
        metrics.completion_times.append(est_cloud_finish - task.arrival_time)
        metrics.total_used_cpu_cycles += task.cpu_demand
        return True, "cloud"

    return False, "rejected"


def install_service_on_edge(
    es: EdgeServer,
    service: Service,
    current_time: float,
    metrics: SimulationMetrics,
    eviction_order: List[int]
):
    """
    Installs a service onto an Edge Server, evicting existing services according to
    `eviction_order` until sufficient storage space is freed. Records replacement costs.
    """
    if service.service_id in es.cached_services:
        return

    for victim_id in eviction_order:
        if es.free_storage(_SERVICE_REGISTRY) >= service.storage_size:
            break
        if victim_id in es.cached_services and victim_id != service.service_id:
            es.cached_services.remove(victim_id)
            es.service_ready_time.pop(victim_id, None)

    if es.free_storage(_SERVICE_REGISTRY) >= service.storage_size:
        es.cached_services.add(service.service_id)
        es.service_ready_time[service.service_id] = current_time + service.retrieval_time
        metrics.replacement_count += 1
        metrics.total_bandwidth_cost += service.bandwidth_cost
        metrics.total_retrieval_delay += service.retrieval_time


_SERVICE_REGISTRY: Dict[int, Service] = {}


def run_baseline_simulation(
    policy: str,
    services: Dict[int, Service],
    workload_by_slot: List[List[Task]],
    num_edges: int = 10,
    num_clouds: int = 3,
    slot_duration: float = 2.0
) -> SimulationMetrics:
    """
    Runs a complete multi-slot simulation for one of the 5 baseline policies:
    'cloud_only', 'static', 'lru', 'lfu', 'greedy'
    """
    global _SERVICE_REGISTRY
    _SERVICE_REGISTRY = services

    edges = [EdgeServer(server_id=i) for i in range(num_edges)]
    clouds = [CloudServer(server_id=i) for i in range(num_clouds)]
    metrics = SimulationMetrics(num_slots=len(workload_by_slot))

    last_used: List[Dict[int, float]] = [{} for _ in range(num_edges)]
    freq_count: List[Dict[int, int]] = [{sid: 0 for sid in services} for _ in range(num_edges)]

    # Initialize cache for edge-enabled baselines using initial slot-0 demand
    if policy in ("static", "greedy", "lru", "lfu"):
        slot0_counts = {}
        for t in workload_by_slot[0]:
            slot0_counts[t.service_id] = slot0_counts.get(t.service_id, 0) + 1
        ranked_init = sorted(services.keys(), key=lambda s: slot0_counts.get(s, 0), reverse=True)

        for e_idx, es in enumerate(edges):
            rotated = ranked_init[e_idx % 5:] + ranked_init[:e_idx % 5]
            for sid in rotated:
                if es.free_storage(services) >= services[sid].storage_size:
                    es.cached_services.add(sid)
                    es.service_ready_time[sid] = 0.0
                    last_used[e_idx][sid] = 0.0

    # Simulate slot by slot
    for slot_idx, slot_tasks in enumerate(workload_by_slot):
        slot_hits_before = metrics.cache_hits
        slot_misses_before = metrics.cache_misses
        slot_acc_before = metrics.accepted_tasks

        slot_demand: Dict[int, int] = {}

        for task in slot_tasks:
            sid = task.service_id
            slot_demand[sid] = slot_demand.get(sid, 0) + 1

            # 1. Offload the task FIRST so cache misses are accurately recorded
            accepted, location = offload_task(task, edges, clouds, metrics)

            # 2. Reactive policies ('lru', 'lfu') fetch missing service AFTER a cache miss
            if policy in ("lru", "lfu"):
                if not any(sid in es.cached_services for es in edges):
                    target_idx = int(np.argmin([es.cpu_free_time for es in edges]))
                    target_es = edges[target_idx]

                    if policy == "lru":
                        victims = sorted(
                            list(target_es.cached_services),
                            key=lambda s: last_used[target_idx].get(s, -1.0)
                        )
                    else:  # lfu
                        victims = sorted(
                            list(target_es.cached_services),
                            key=lambda s: freq_count[target_idx].get(s, 0)
                        )

                    install_service_on_edge(target_es, services[sid], task.arrival_time, metrics, victims)
                    last_used[target_idx][sid] = task.arrival_time

            # Update usage stats for edges holding this service
            for e_idx, es in enumerate(edges):
                if sid in es.cached_services:
                    last_used[e_idx][sid] = task.arrival_time
                    freq_count[e_idx][sid] += 1

        # End-of-slot update for 'greedy' (Popularity-Based Caching ignoring replacement cost)
        if policy == "greedy" and slot_idx < len(workload_by_slot) - 1:
            ranked_slot = sorted(services.keys(), key=lambda s: slot_demand.get(s, 0), reverse=True)
            slot_end = (slot_idx + 1) * slot_duration

            for e_idx, es in enumerate(edges):
                desired = []
                used_mem = 0.0
                rotated = ranked_slot[e_idx % 4:] + ranked_slot[:e_idx % 4]
                for sid in rotated:
                    if used_mem + services[sid].storage_size <= es.storage_capacity:
                        desired.append(sid)
                        used_mem += services[sid].storage_size

                victims = sorted(
                    [s for s in es.cached_services if s not in desired],
                    key=lambda s: slot_demand.get(s, 0)
                )
                for sid in desired:
                    if sid not in es.cached_services:
                        install_service_on_edge(es, services[sid], slot_end, metrics, victims)

        # Record per-slot hit ratio and acceptance ratio
        s_hits = metrics.cache_hits - slot_hits_before
        s_misses = metrics.cache_misses - slot_misses_before
        s_total = s_hits + s_misses
        metrics.slot_hit_ratios.append((s_hits / s_total * 100.0) if s_total > 0 else 0.0)
        metrics.slot_accept_ratios.append(
            ((metrics.accepted_tasks - slot_acc_before) / len(slot_tasks) * 100.0) if slot_tasks else 0.0
        )

    return metrics


if __name__ == "__main__":
    from workload import generate_services, generate_dynamic_workload

    services = generate_services(num_services=100, seed=42)
    workload = generate_dynamic_workload(scenario="sudden_spike", seed=42)

    for pol in ["cloud_only", "static", "lru", "lfu", "greedy"]:
        m = run_baseline_simulation(pol, services, workload)
        print(f"\n--- {pol.upper()} ---")
        for k, v in m.summary(10, 20.0, 3, 5.0, 100.0).items():
            print(f"  {k}: {v}")