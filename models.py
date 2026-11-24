import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class Service:
    service_id: int
    storage_size: float      # GB (40 to 80 GB in base paper)
    retrieval_time: float    # Seconds to fetch/deploy onto an Edge Server
    bandwidth_cost: float    # Cost units to download from Cloud to Edge

@dataclass
class Task:
    task_id: int
    service_id: int
    arrival_time: float      # Time slot / second task arrives
    cpu_demand: float        # GHz-sec (0.5 to 1.2 in base paper)
    input_data: float        # MB (5 to 6 MB in base paper)
    output_data: float       # MB (small output, e.g., 0.5 MB)
    deadline: float          # Relative deadline in seconds (1 to 5s in base paper)

@dataclass
class EdgeServer:
    server_id: int
    cpu_capacity: float = 20.0       # GHz (Base paper: 20 GHz)
    storage_capacity: float = 300.0  # GB (Base paper: 300 GB)
    num_channels: int = 10           # Channels (Base paper: 10)
    bandwidth: float = 60.0          # Mbps per channel (Base paper: 60 Mbps)
    
    # Dynamic state tracking
    cached_services: set = field(default_factory=set)
    channel_free_time: List[float] = field(default_factory=list)
    cpu_free_time: float = 0.0
    service_ready_time: Dict[int, float] = field(default_factory=dict)

    def __post_init__(self):
        self.channel_free_time = [0.0] * self.num_channels

    def used_storage(self, services: Dict[int, Service]) -> float:
        return sum(services[sid].storage_size for sid in self.cached_services)

    def free_storage(self, services: Dict[int, Service]) -> float:
        return self.storage_capacity - self.used_storage(services)

    def estimate_finish_time(self, task: Task) -> tuple[float, int]:
        """
        Implements Eq. (2)-(8) from Gao et al. (2022):
        Computes task finish time considering both network and CPU bottlenecks.
        Returns (finish_time, best_channel_index).
        """
        # Convert MB to Megabits (1 MB = 8 Mb) for Mbps calculation
        input_bits = task.input_data * 8.0
        output_bits = task.output_data * 8.0

        # Pick the communication channel that becomes free earliest
        best_ch = int(np.argmin(self.channel_free_time))
        net_start = max(task.arrival_time, self.channel_free_time[best_ch])
        net_finish = net_start + (input_bits / self.bandwidth)

        # Service cannot run until input arrives, CPU is free, AND service is done downloading
        ready_time = self.service_ready_time.get(task.service_id, 0.0)
        cpu_start = max(net_finish, self.cpu_free_time, ready_time)
        cpu_finish = cpu_start + (task.cpu_demand / self.cpu_capacity)

        # Output transmission time
        total_finish = cpu_finish + (output_bits / self.bandwidth)
        return total_finish, best_ch

    def assign_task(self, task: Task) -> bool:
        """Executes task on this Edge Server if cached and deadline is met."""
        if task.service_id not in self.cached_services:
            return False

        finish_time, best_ch = self.estimate_finish_time(task)
        if finish_time <= (task.arrival_time + task.deadline):
            input_bits = task.input_data * 8.0
            output_bits = task.output_data * 8.0
            
            net_start = max(task.arrival_time, self.channel_free_time[best_ch])
            net_finish = net_start + (input_bits / self.bandwidth)
            
            ready_time = self.service_ready_time.get(task.service_id, 0.0)
            cpu_start = max(net_finish, self.cpu_free_time, ready_time)
            cpu_finish = cpu_start + (task.cpu_demand / self.cpu_capacity)

            self.channel_free_time[best_ch] = net_finish
            self.cpu_free_time = cpu_finish
            return True
        return False


@dataclass
class CloudServer:
    server_id: int
    cpu_capacity: float = 5.0    # GHz (Base paper: 5 GHz)
    bandwidth: float = 15.0      # Mbps (Base paper: 15 Mbps)
    channel_free_time: float = 0.0
    cpu_free_time: float = 0.0

    def estimate_finish_time(self, task: Task) -> float:
        """Cloud stores all services, 1 channel per CS (Base paper Sec II-A)."""
        input_bits = task.input_data * 8.0
        output_bits = task.output_data * 8.0

        net_start = max(task.arrival_time, self.channel_free_time)
        net_finish = net_start + (input_bits / self.bandwidth)

        cpu_start = max(net_finish, self.cpu_free_time)
        cpu_finish = cpu_start + (task.cpu_demand / self.cpu_capacity)

        return cpu_finish + (output_bits / self.bandwidth)

    def assign_task(self, task: Task) -> bool:
        finish_time = self.estimate_finish_time(task)
        if finish_time <= (task.arrival_time + task.deadline):
            input_bits = task.input_data * 8.0
            net_start = max(task.arrival_time, self.channel_free_time)
            net_finish = net_start + (input_bits / self.bandwidth)
            
            cpu_start = max(net_finish, self.cpu_free_time)
            cpu_finish = cpu_start + (task.cpu_demand / self.cpu_capacity)

            self.channel_free_time = net_finish
            self.cpu_free_time = cpu_finish
            return True
        return False