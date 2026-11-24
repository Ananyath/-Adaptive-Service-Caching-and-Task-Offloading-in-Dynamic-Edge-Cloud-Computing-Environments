import numpy as np
from typing import Dict, List
from models import Service, Task, EdgeServer, CloudServer
from baselines import SimulationMetrics, offload_task


class CostAwareAdaptiveCache:
    """
    Proposed Cost-Aware Adaptive Service Caching & Task Offloading Controller.
    Balances EWMA demand estimation against retrieval delay and bandwidth replacement costs.
    """
    def __init__(
        self,
        services: Dict[int, Service],
        decay_factor: float = 0.65,
        cost_weight: float = 0.015,
        horizon_slots: int = 6
    ):
        self.services = services
        self.decay_factor = decay_factor
        self.cost_weight = cost_weight
        self.horizon_slots = horizon_slots
        self.ewma_demand: Dict[int, float] = {sid: 0.0 for sid in services}

    def update_demand(self, slot_counts: Dict[int, int]):
        """Updates exponentially weighted moving average demand at the end of a time slot."""
        for sid in self.services:
            current = float(slot_counts.get(sid, 0))
            self.ewma_demand[sid] = (
                self.decay_factor * self.ewma_demand[sid]
                + (1.0 - self.decay_factor) * current
            )

    def service_benefit_density(self, sid: int, replica_count: int) -> float:
        """
        Estimates per-GB value of caching service `sid` over the lookahead horizon.
        Accounts for network/compute latency saved vs Cloud and edge-cluster replication.
        """
        svc = self.services[sid]
        predicted_requests = self.ewma_demand[sid] * self.horizon_slots
        replication_factor = 1.0 / (1.0 + 0.35 * replica_count)
        avg_latency_saved = 0.45
        total_benefit = predicted_requests * avg_latency_saved * replication_factor
        return total_benefit / svc.storage_size

    def replacement_penalty_density(self, sid: int) -> float:
        """Computes normalized replacement overhead (bandwidth + retrieval downtime) per GB."""
        svc = self.services[sid]
        raw_cost = svc.bandwidth_cost + (svc.retrieval_time * 10.0)
        return (self.cost_weight * raw_cost) / svc.storage_size


def run_adaptive_simulation(
    services: Dict[int, Service],
    workload_by_slot: List[List[Task]],
    num_edges: int = 10,
    num_clouds: int = 3,
    slot_duration: float = 2.0,
    decay_factor: float = 0.65,
    cost_weight: float = 0.015
) -> SimulationMetrics:
    """
    Runs the complete simulation using the proposed Cost-Aware Adaptive Framework.
    """
    edges = [EdgeServer(server_id=i) for i in range(num_edges)]
    clouds = [CloudServer(server_id=i) for i in range(num_clouds)]
    metrics = SimulationMetrics(num_slots=len(workload_by_slot))

    controller = CostAwareAdaptiveCache(
        services=services,
        decay_factor=decay_factor,
        cost_weight=cost_weight
    )

    # 1. Warm-start cache using initial slot-0 demand distribution
    slot0_counts: Dict[int, int] = {}
    for t in workload_by_slot[0]:
        slot0_counts[t.service_id] = slot0_counts.get(t.service_id, 0) + 1
    controller.update_demand(slot0_counts)

    ranked_init = sorted(services.keys(), key=lambda s: controller.ewma_demand[s], reverse=True)
    for e_idx, es in enumerate(edges):
        rotated = ranked_init[e_idx % 5:] + ranked_init[:e_idx % 5]
        for sid in rotated:
            if es.free_storage(services) >= services[sid].storage_size:
                es.cached_services.add(sid)
                es.service_ready_time[sid] = 0.0

    # 2. Simulate time-slotted execution + cost-aware adaptive cache updates
    for slot_idx, slot_tasks in enumerate(workload_by_slot):
        slot_hits_before = metrics.cache_hits
        slot_misses_before = metrics.cache_misses
        slot_acc_before = metrics.accepted_tasks

        slot_demand: Dict[int, int] = {}

        for task in slot_tasks:
            sid = task.service_id
            slot_demand[sid] = slot_demand.get(sid, 0) + 1
            offload_task(task, edges, clouds, metrics)

        controller.update_demand(slot_demand)

        # Perform Cost-Aware Adaptive Cache Replacement at slot boundary
        if slot_idx < len(workload_by_slot) - 1:
            slot_end = (slot_idx + 1) * slot_duration

            replica_counts = {
                sid: sum(1 for es in edges if sid in es.cached_services)
                for sid in services
            }

            for es in edges:
                candidates = [
                    sid for sid in services
                    if sid not in es.cached_services and controller.ewma_demand[sid] > 0.05
                ]
                candidates.sort(
                    key=lambda s: (
                        controller.service_benefit_density(s, replica_counts[s])
                        - controller.replacement_penalty_density(s)
                    ),
                    reverse=True
                )

                for cand_id in candidates[:10]:
                    cand_svc = services[cand_id]
                    cand_net_value = (
                        controller.service_benefit_density(cand_id, replica_counts[cand_id])
                        - controller.replacement_penalty_density(cand_id)
                    )

                    if es.free_storage(services) >= cand_svc.storage_size:
                        if cand_net_value > 0:
                            es.cached_services.add(cand_id)
                            es.service_ready_time[cand_id] = slot_end + cand_svc.retrieval_time
                            replica_counts[cand_id] += 1
                            metrics.replacement_count += 1
                            metrics.total_bandwidth_cost += cand_svc.bandwidth_cost
                            metrics.total_retrieval_delay += cand_svc.retrieval_time
                        continue

                    victims = sorted(
                        list(es.cached_services),
                        key=lambda v: controller.service_benefit_density(v, max(0, replica_counts[v] - 1))
                    )

                    freed_space = es.free_storage(services)
                    victim_pool = []
                    victim_value_sum = 0.0

                    for v_id in victims:
                        if freed_space >= cand_svc.storage_size:
                            break
                        victim_pool.append(v_id)
                        freed_space += services[v_id].storage_size
                        victim_value_sum += controller.service_benefit_density(
                            v_id, max(0, replica_counts[v_id] - 1)
                        )

                    if freed_space >= cand_svc.storage_size and cand_net_value > victim_value_sum:
                        for v_id in victim_pool:
                            es.cached_services.remove(v_id)
                            es.service_ready_time.pop(v_id, None)
                            replica_counts[v_id] -= 1

                        es.cached_services.add(cand_id)
                        es.service_ready_time[cand_id] = slot_end + cand_svc.retrieval_time
                        replica_counts[cand_id] += 1

                        metrics.replacement_count += 1
                        metrics.total_bandwidth_cost += cand_svc.bandwidth_cost
                        metrics.total_retrieval_delay += cand_svc.retrieval_time

        s_hits = metrics.cache_hits - slot_hits_before
        s_misses = metrics.cache_misses - slot_misses_before
        s_total = s_hits + s_misses
        metrics.slot_hit_ratios.append((s_hits / s_total * 100.0) if s_total > 0 else 0.0)
        metrics.slot_accept_ratios.append(
            ((metrics.accepted_tasks - slot_acc_before) / len(slot_tasks) * 100.0) if slot_tasks else 0.0
        )

    return metrics