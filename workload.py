import numpy as np
from typing import Dict, List
from models import Service, Task

def generate_services(num_services: int = 100, seed: int = 42) -> Dict[int, Service]:
    """
    Creates S services using parameters from Gao et al. (2022):
    - Storage size: [40, 80] GB
    - Retrieval time & bandwidth cost: proportional to storage size (for replacement overhead)
    """
    rng = np.random.default_rng(seed)
    services = {}
    for sid in range(num_services):
        storage = float(rng.uniform(40.0, 80.0))
        # Assume backhaul link transfers service image at ~400 Mbps (50 MB/s -> ~0.05s per GB)
        retrieval_time = round(storage * 0.04, 3)  # ~1.6s to 3.2s downtime/fetch delay
        bandwidth_cost = round(storage * 1.0, 2)   # Cost proportional to GB downloaded
        services[sid] = Service(
            service_id=sid,
            storage_size=round(storage, 2),
            retrieval_time=retrieval_time,
            bandwidth_cost=bandwidth_cost
        )
    return services


def _zipf_probabilities(num_services: int, alpha: float = 1.0) -> np.ndarray:
    """Generates normalized Zipf probability distribution across ranked services."""
    ranks = np.arange(1, num_services + 1, dtype=float)
    weights = 1.0 / np.power(ranks, alpha)
    return weights / np.sum(weights)


def generate_dynamic_workload(
    num_services: int = 100,
    num_slots: int = 50,
    tasks_per_slot: int = 40,
    slot_duration: float = 2.0,
    scenario: str = "sudden_spike",
    zipf_alpha: float = 1.0,
    seed: int = 42
) -> List[List[Task]]:
    """
    Generates time-slotted task arrivals under three popularity scenarios:
    - 'static': Fixed service popularity ranking throughout all slots.
    - 'gradual': Service popularity ranking slowly shifts every slot.
    - 'sudden_spike': Popularity ranking abruptly inverts/shifts at the halfway mark.
    """
    rng = np.random.default_rng(seed)
    base_probs = _zipf_probabilities(num_services, alpha=zipf_alpha)
    
    # Initial ranking of service IDs (0 is initially most popular)
    ranking = np.arange(num_services)
    
    workload_by_slot: List[List[Task]] = []
    task_counter = 0

    for slot in range(num_slots):
        # 1. Update service popularity ranking based on scenario
        if scenario == "gradual":
            # Every slot, swap a few adjacent ranks to simulate gradual drift
            for _ in range(5):
                idx = int(rng.integers(0, num_services - 1))
                ranking[idx], ranking[idx + 1] = ranking[idx + 1], ranking[idx]

        elif scenario == "sudden_spike":
            # At halfway point (and 75% mark), trigger an abrupt demand shift
            if slot == num_slots // 2:
                # Reverse top 30 services with bottom 30 services
                ranking = np.roll(ranking, shift=30)
            elif slot == (3 * num_slots) // 4:
                ranking = np.roll(ranking, shift=30)

        # Map Zipf probabilities to the current service ranking
        slot_probs = np.zeros(num_services)
        for rank_idx, service_id in enumerate(ranking):
            slot_probs[service_id] = base_probs[rank_idx]

        # 2. Generate tasks arriving during this time slot
        slot_start_time = slot * slot_duration
        slot_tasks: List[Task] = []

        # Sample which services are requested in this slot
        chosen_services = rng.choice(num_services, size=tasks_per_slot, p=slot_probs)
        # Random arrival timestamps within the current time slot
        arrival_offsets = np.sort(rng.uniform(0.0, slot_duration, size=tasks_per_slot))

        for i in range(tasks_per_slot):
            # Task parameters matching Gao et al. (2022) Section IV-A:
            # CPU: [0.5, 1.2] GHz, Input: [5, 6] MB, Deadline: [1, 5] seconds
            cpu_demand = float(rng.uniform(0.5, 1.2))
            input_data = float(rng.uniform(5.0, 6.0))
            output_data = float(rng.uniform(0.2, 0.5))
            deadline = float(rng.uniform(1.0, 5.0))

            task = Task(
                task_id=task_counter,
                service_id=int(chosen_services[i]),
                arrival_time=round(slot_start_time + float(arrival_offsets[i]), 4),
                cpu_demand=round(cpu_demand, 3),
                input_data=round(input_data, 3),
                output_data=round(output_data, 3),
                deadline=round(deadline, 3)
            )
            slot_tasks.append(task)
            task_counter += 1

        workload_by_slot.append(slot_tasks)

    return workload_by_slot


# Quick test block to verify everything runs cleanly
if __name__ == "__main__":
    services = generate_services(num_services=100, seed=42)
    workload = generate_dynamic_workload(scenario="sudden_spike", seed=42)
    print(f"Generated {len(services)} services. Sample Service 0: {services[0]}")
    print(f"Generated {len(workload)} slots with {len(workload[0])} tasks/slot.")
    print(f"Slot 0 top requested service: {workload[0][0].service_id}")
    print(f"Slot 25 (post-spike) sample task: {workload[25][0]}")