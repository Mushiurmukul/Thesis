
import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mec_system.env.mec_env import MECEnv
from mec_system.config import *

def print_all_action_spaces():
    print("="*80)
    print("ACTION SPACE DETAILS FOR ALL TASKS")
    print("="*80)
    
    env = MECEnv()
    
    print("\n1. GLOBAL ACTION SPACE (Environment Level)")
    print("-" * 80)
    print(f"   Total Devices in System: {NUM_EDGE_SERVERS * NUM_DEVICES_PER_SERVER}")
    print(f"   Action Space Shape: {env.action_space.shape}")
    print(f"   Action Space Type: {type(env.action_space).__name__}")
    print(f"   Action Range: [{env.action_space.low[0][0]}, {env.action_space.high[0][0]}]")
    
    print("\n2. PER-AGENT ACTION SPACE (Edge Server Level)")
    print("-" * 80)
    num_neighbors = NUM_EDGE_SERVERS - 1
    dim_per_device = 1 + num_neighbors
    total_act_dim_per_agent = NUM_DEVICES_PER_SERVER * dim_per_device
    
    for agent_id in range(NUM_EDGE_SERVERS):
        print(f"\n   Agent {agent_id} (Edge Server {agent_id}):")
        print(f"   - Manages Devices: {agent_id * NUM_DEVICES_PER_SERVER} to {(agent_id + 1) * NUM_DEVICES_PER_SERVER - 1}")
        print(f"   - Action Dimension: {total_act_dim_per_agent} (flattened)")
        print(f"   - Structure: [{NUM_DEVICES_PER_SERVER} devices × {dim_per_device} components]")
    
    print("\n3. PER-DEVICE ACTION SPACE")
    print("-" * 80)
    print(f"   Each device has {dim_per_device} action components:")
    print(f"   - Alpha (1): Local execution fraction [0, 1]")
    print(f"   - Mu_1 to Mu_{num_neighbors} ({num_neighbors}): Neighbor offload fractions [0, 1]")
    
    print("\n4. DETAILED ACTION BREAKDOWN FOR ALL DEVICES")
    print("-" * 80)
    
    for server_id in range(NUM_EDGE_SERVERS):
        print(f"\n   Edge Server {server_id}:")
        neighbor_ids = [k for k in range(NUM_EDGE_SERVERS) if k != server_id]
        
        for local_dev_id in range(NUM_DEVICES_PER_SERVER):
            global_dev_id = server_id * NUM_DEVICES_PER_SERVER + local_dev_id
            start_idx = local_dev_id * dim_per_device
            end_idx = start_idx + dim_per_device
            
            print(f"      Device {global_dev_id} (Local ID: {local_dev_id}):")
            print(f"         Action indices in agent vector: [{start_idx}:{end_idx}]")
            print(f"         Components:")
            print(f"            [{start_idx}] Alpha: Local execution fraction")
            for i, neighbor_id in enumerate(neighbor_ids):
                print(f"            [{start_idx + 1 + i}] Mu_{i+1}: Offload fraction to Edge Server {neighbor_id}")
    
    print("\n5. ACTION INTERPRETATION")
    print("-" * 80)
    print("   For a device with action [Alpha, Mu_1, Mu_2, ...]:")
    print("   - Local execution:        Alpha × Task")
    print("   - Offload to serving edge: (1 - Alpha) × (1 - Σ Mu_i) × Task")
    print("   - Offload to neighbor k:   (1 - Alpha) × Mu_k × Task")
    print("   - Constraint: 0 ≤ Alpha ≤ 1, 0 ≤ Mu_i ≤ 1, Σ Mu_i ≤ 1")
    
    print("\n6. SAMPLE ACTION VECTOR (Agent 0)")
    print("-" * 80)
    sample_action = np.random.random(total_act_dim_per_agent)
    print(f"   Shape: ({total_act_dim_per_agent},)")
    print(f"   Sample values (first 9 components):")
    for i in range(min(9, total_act_dim_per_agent)):
        dev_id = i // dim_per_device
        comp_id = i % dim_per_device
        if comp_id == 0:
            print(f"      [{i}] Device {dev_id} Alpha = {sample_action[i]:.4f}")
        else:
            print(f"      [{i}] Device {dev_id} Mu_{comp_id} = {sample_action[i]:.4f}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    print_all_action_spaces()
