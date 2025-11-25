
import numpy as np
from mec_system.config import *

class Task:
    def __init__(self, task_id, creation_time):
        self.task_id = task_id
        self.creation_time = creation_time
        
        # Randomly generate task characteristics based on config ranges
        self.data_size = np.random.uniform(TASK_DATA_SIZE_MIN, TASK_DATA_SIZE_MAX)
        self.cpu_cycles_per_bit = np.random.uniform(TASK_CPU_CYCLES_PER_BIT_MIN, TASK_CPU_CYCLES_PER_BIT_MAX)
        self.deadline = np.random.uniform(TASK_DEADLINE_MIN, TASK_DEADLINE_MAX)
        
        self.total_cpu_cycles = self.data_size * self.cpu_cycles_per_bit
        self.vulnerability = np.random.uniform(TASK_VULNERABILITY_MIN, TASK_VULNERABILITY_MAX)

class MobileDevice:
    def __init__(self, device_id, associated_edge_id):
        self.device_id = device_id
        self.associated_edge_id = associated_edge_id
        self.cpu_capacity = DEVICE_CPU_CAPACITY
        
    def generate_task(self, time_step):
        # Simple task generation logic (can be made more complex, e.g., Poisson process)
        return Task(task_id=f"{self.device_id}_{time_step}", creation_time=time_step)

class EdgeServer:
    def __init__(self, server_id):
        self.server_id = server_id
        self.cpu_capacity = EDGE_CPU_CAPACITY
        self.bandwidth_capacity = EDGE_BANDWIDTH_CAPACITY
        self.storage_capacity = EDGE_STORAGE_CAPACITY
        
        # M/M/1/K Queue state
        self.queue = [] # List of tasks
        self.max_queue_size = MAX_QUEUE_SIZE
        
        # Current Load (for observation)
        self.current_cpu_load = 0.0
        self.current_bw_load = 0.0
        self.current_storage_load = 0.0
        
        # Stats
        self.arrival_rate = 0.0 # lambda
        self.service_rate = self.cpu_capacity / ((TASK_CPU_CYCLES_PER_BIT_MIN + TASK_CPU_CYCLES_PER_BIT_MAX) / 2 * (TASK_DATA_SIZE_MIN + TASK_DATA_SIZE_MAX) / 2) # approx mu
        
    def update_load(self):
        # Simplified load calculation based on queue length or active tasks
        # In a real simulation, this would track actual resource usage
        if len(self.queue) > 0:
             self.current_cpu_load = min(1.0, sum([t.total_cpu_cycles for t in self.queue]) / self.cpu_capacity)
        else:
            self.current_cpu_load = 0.0
            
    def get_queue_length(self):
        return len(self.queue)
