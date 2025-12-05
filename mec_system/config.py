from dataclasses import dataclass

@dataclass
class SystemConfig:
    NUM_EDGE_SERVERS: int = 5
    NUM_DEVICES_PER_SERVER: int = 20  # Reduced to make system stable (9 tasks/sec total)
    NUM_TIME_STEPS: int = 100  # Reduced for faster 1000-episode run
    TIME_SLOT_DURATION: float = 7 # seconds (Increased to allow queue clearing)
    
    # Resource Capacities (Based on 2024-2025 Research)
    # Local MEC Server: 3-5 GHz per core, 10-20 GHz total (Liu et al. 2025)
    # Using mid-range: 15 GHz total (e.g., 4-core @ 3.75 GHz)
    EDGE_CPU_CAPACITY: float = 24e9     # 10 GHz (mid-range)
    # Local Device (IoT/Phone): 0.5-1.0 GHz (Iqbal et al. 2025)
    # Using upper range for modern smartphones
    DEVICE_CPU_CAPACITY: float = 1e9    # 1 GHz
    EDGE_BANDWIDTH_CAPACITY: float = 1e9 # 1 Gbps
    EDGE_STORAGE_CAPACITY: float = 50e9  # 50 GB
    MAX_QUEUE_SIZE: int = 20  # Reduced to make congestion more apparent

@dataclass
class TaskConfig:
    # Task Characteristics (Adjusted for Solvability)
    # Input Data Size: 0.2 MB - 0.8 MB (smaller, more realistic tasks)
    # Small tasks (0.2 MB) can be processed locally or offloaded quickly
    # Large tasks (0.8 MB) are challenging but solvable within 1s with good network
    DATA_SIZE_MIN: float = 0.5e6   # 0.5 MB
    DATA_SIZE_MAX: float = 5e6   # 5 MB
    # CPU Cycles per Bit: 500-2500 cycles/bit (Jin et al. 2024)
    # 500 for simple processing, 2500 for DNN inference
    CPU_CYCLES_PER_BIT_MIN: int = 500
    CPU_CYCLES_PER_BIT_MAX: int = 2500
    # Task Deadline: 0.1s - 1.5s (Relaxed for convergence)
    DEADLINE_MIN: float = 0.1      # 100ms (strict URLLC)
    DEADLINE_MAX: float = 2.5      # 2.5 seconds (Relaxed)
    RECOM_TIME: float = 0.01

@dataclass
class NetworkConfig:
    # Communication (V2I / Device-to-Edge 5G)
    # Transmission Rate: 20-100 Mbps (Avoids 10 Mbps bottleneck)
    UPLINK_RATE_MIN: float = 20e6   # 20 Mbps (Relaxed)
    UPLINK_RATE_MAX: float = 100e6  # 100 Mbps
    DOWNLINK_RATE_MIN: float = 50e6 # 50 Mbps
    DOWNLINK_RATE_MAX: float = 200e6 # 200 Mbps
    # Neighboring server: high-speed fiber backhaul
    INTER_EDGE_RATE_MIN: float = 1e9 # 1 Gbps
    INTER_EDGE_RATE_MAX: float = 10e9 # 10 Gbps

@dataclass
class ReliabilityConfig:
    # Reliability & Fault Parameters (2024-2025 Research)
    # Channel Error Rate: 10^-5 to 10^-3 (PER in 5G/6G)
    CHANNEL_ERROR_RATE_MIN: float = 1e-5
    CHANNEL_ERROR_RATE_MAX: float = 1e-3
    # Processor Fault Rate: 10^-4 to 10^-3 per second
    PROCESSOR_FAULT_RATE_MIN: float = 1e-4
    PROCESSOR_FAULT_RATE_MAX: float = 1e-3
    # Task Vulnerability Factor: 0.001 - 0.01
    TASK_VULNERABILITY_MIN: float = 0.001
    TASK_VULNERABILITY_MAX: float = 0.01
    QUEUE_SENSITIVITY: float = 0.05

@dataclass
class TrainingConfig:
    # Optimization Hyperparameters
    BETA: float = 0.5
    
    # Constraints (Research-based thresholds)
    # Fairness Threshold: 0.7-0.8 (Jain's Index)
    FAIRNESS_THRESHOLD: float = 0.80
    # Reliability Threshold: 95%-99.9% (URLLC targets 99.999%, general MEC ~99%)
    RELIABILITY_THRESHOLD: float = 0.80
    
    # PPO Parameters
    LR_ACTOR: float = 1.5e-4
    LR_CRITIC: float = 6e-4
    LR_DUAL: float = 1e-5
    GAMMA: float = 0.99
    PPO_CLIP_EPSILON: float = 0.2
    PPO_EPOCHS: int = 12
    BATCH_SIZE: int = 256
    ENTROPY_COEF: float = 0.001  # Reduced for Beta distribution
    
    # Dual Ascent
    MAX_LAMBDA: float = 5.0

# Global Configuration Instance
class Config:
    system = SystemConfig()
    task = TaskConfig()
    network = NetworkConfig()
    reliability = ReliabilityConfig()
    training = TrainingConfig()
