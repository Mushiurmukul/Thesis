
import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mec_system.env.mec_env import MECEnv
from mec_system.config import *

def print_action_space():
    print("Initializing MEC Environment...")
    env = MECEnv()
    
    print("\n" + "="*40)
    print("ACTION SPACE DETAILS")
    print("="*40)
    
    # Check if it has a gym-style action_space
    if hasattr(env, 'action_space'):
        print(f"Action Space Object: {env.action_space}")
        if hasattr(env.action_space, 'shape'):
            print(f"Action Space Shape: {env.action_space.shape}")
        if hasattr(env.action_space, 'high'):
            print(f"Action High: {env.action_space.high}")
        if hasattr(env.action_space, 'low'):
            print(f"Action Low: {env.action_space.low}")
    else:
        print("No standard 'action_space' attribute found.")
        
    print("-" * 40)
    print("Configuration derived dimensions:")
    print(f"NUM_EDGE_SERVERS: {NUM_EDGE_SERVERS}")
    print(f"NUM_DEVICES_PER_SERVER: {NUM_DEVICES_PER_SERVER}")
    
    # Based on previous context: Action dim per server = Num_Devices * (1 + Neighbors)
    num_neighbors = NUM_EDGE_SERVERS - 1
    dim_per_device = 1 + num_neighbors
    total_act_dim = NUM_DEVICES_PER_SERVER * dim_per_device
    
    print(f"Neighbors per server: {num_neighbors}")
    print(f"Action components per device: {dim_per_device} (1 alpha + {num_neighbors} mus)")
    print(f"Total Action Dimension per Agent (Server): {total_act_dim}")
    print("-" * 40)
    
    print("\nSample Action Structure (for one agent):")
    print(f"Shape: ({total_act_dim},)")
    print("Interpretation:")
    print(f"  [Device 1 Alpha, Device 1 Mu_1, ..., Device 1 Mu_{num_neighbors},")
    print(f"   Device 2 Alpha, ... ]")
    
if __name__ == "__main__":
    print_action_space()
