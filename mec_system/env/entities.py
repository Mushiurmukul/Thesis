
import numpy as np
from mec_system.config import Config

class Task:
    def __init__(self, task_id, creation_time):
        self.task_id = task_id
        self.creation_time = creation_time
        
        # Randomly generate task characteristics based on config ranges
        self.data_size = np.random.uniform(Config.task.DATA_SIZE_MIN, Config.task.DATA_SIZE_MAX)
        self.result_size = self.data_size * 0.1 # Assume result is 10% of input size
        self.cpu_cycles_per_bit = np.random.uniform(Config.task.CPU_CYCLES_PER_BIT_MIN, Config.task.CPU_CYCLES_PER_BIT_MAX)
        
        # Deadline: uniform range 2.0 to 2.5 seconds
        self.deadline = np.random.uniform(2.0, 2.5)
            
        self.total_cpu_cycles = self.data_size * self.cpu_cycles_per_bit
        self.vulnerability = np.random.uniform(Config.reliability.TASK_VULNERABILITY_MIN, Config.reliability.TASK_VULNERABILITY_MAX)

class MobileDevice:
    def __init__(self, device_id, associated_edge_id):
        self.device_id = device_id
        self.associated_edge_id = associated_edge_id
        self.cpu_capacity = Config.system.DEVICE_CPU_CAPACITY
        
    def generate_task(self, time_step):
        # Simple task generation logic (can be made more complex, e.g., Poisson process)
        return Task(task_id=f"{self.device_id}_{time_step}", creation_time=time_step)

class EdgeServer:
    def __init__(self, server_id):
        self.server_id = server_id
        self.cpu_capacity = Config.system.EDGE_CPU_CAPACITY
        self.bandwidth_capacity = Config.system.EDGE_BANDWIDTH_CAPACITY
        self.storage_capacity = Config.system.EDGE_STORAGE_CAPACITY
        
        # M/M/1/K Queue state
        self.queue = [] # List of tasks
        self.max_queue_size = Config.system.MAX_QUEUE_SIZE
        
        # Current Load (for observation)
        self.current_cpu_load = 0.0
        self.current_bw_load = 0.0
        self.current_storage_load = 0.0
        
        # Stats
        self.arrival_rate = 0.0 # lambda
        self.service_rate = self.cpu_capacity / ((Config.task.CPU_CYCLES_PER_BIT_MIN + Config.task.CPU_CYCLES_PER_BIT_MAX) / 2 * (Config.task.DATA_SIZE_MIN + Config.task.DATA_SIZE_MAX) / 2) # approx mu
        
    def update_load(self):
        # Simplified load calculation based on queue length or active tasks
        # In a real simulation, this would track actual resource usage
        if len(self.queue) > 0:
             self.current_cpu_load = min(1.0, sum([t.total_cpu_cycles for t in self.queue]) / self.cpu_capacity)
        else:
            self.current_cpu_load = 0.0
            
    def get_queue_length(self):
        return len(self.queue)
