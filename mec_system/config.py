
# System Configuration

# Simulation Parameters
NUM_EDGE_SERVERS = 3
NUM_DEVICES_PER_SERVER = 5
NUM_TIME_STEPS = 100  # T
TIME_SLOT_DURATION = 1 # seconds

# Resource Capacities (Example values, need tuning)
EDGE_CPU_CAPACITY = 10e9  # 10 GHz (cycles/s)
DEVICE_CPU_CAPACITY = 1e9 # 1 GHz
EDGE_BANDWIDTH_CAPACITY = 100e6 # 100 Mbps
EDGE_STORAGE_CAPACITY = 1e9 # 1 GB

# Task Characteristics (Ranges)
TASK_DATA_SIZE_MIN = 1e5 # bits
TASK_DATA_SIZE_MAX = 1e6 # bits
TASK_CPU_CYCLES_PER_BIT_MIN = 500
TASK_CPU_CYCLES_PER_BIT_MAX = 1500
TASK_DEADLINE_MIN = 0.5 # seconds
TASK_DEADLINE_MAX = 2.0 # seconds

# Communication (Ranges)
UPLINK_RATE_MIN = 5e6 # 5 Mbps
UPLINK_RATE_MAX = 20e6 # 20 Mbps
DOWNLINK_RATE_MIN = 10e6
DOWNLINK_RATE_MAX = 50e6
INTER_EDGE_RATE_MIN = 50e6
INTER_EDGE_RATE_MAX = 100e6

# Reliability Parameters
CHANNEL_ERROR_RATE_MIN = 1e-6
CHANNEL_ERROR_RATE_MAX = 1e-3
PROCESSOR_FAULT_RATE_MIN = 1e-8
PROCESSOR_FAULT_RATE_MAX = 1e-5
TASK_VULNERABILITY_MIN = 0.1
TASK_VULNERABILITY_MAX = 1.0
QUEUE_SENSITIVITY = 0.05

# Queue Parameters
MAX_QUEUE_SIZE = 50 # K

# Optimization Hyperparameters
BETA = 0.5 # Balance between reliability and latency (0 = latency only, 1 = reliability only)
FAIRNESS_THRESHOLD = 0.8 # Jain's Index target
RELIABILITY_THRESHOLD = 0.95
LEARNING_RATE_ACTOR = 1.5e-4  # Further reduced for ultra-smooth convergence
LEARNING_RATE_CRITIC = 6e-4  # Further reduced for ultra-smooth convergence
LEARNING_RATE_DUAL = 2e-4 # Further reduced for minimal oscillation
GAMMA = 0.99 # Discount factor
PPO_CLIP_EPSILON = 0.2
PPO_EPOCHS = 12  # Balanced for convergence without overtraining
BATCH_SIZE = 128  # Increased for more stable gradient estimates
