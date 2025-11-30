
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mec_system.env.mec_env import MECEnv
from mec_system.agents.mappo import MAPPOAgent
from mec_system.config import *

def print_clipped_actions():
    env = MECEnv()
    
    agents = []
    for i in range(NUM_EDGE_SERVERS):
        act_dim = NUM_DEVICES_PER_SERVER * 2
        agents.append(MAPPOAgent(env.obs_space_dim, act_dim))
    
    print("="*80)
    print("ALPHA AND MU FOR ALL TASKS (AFTER CLIPPING TO [0,1])")
    print("="*80)
    
    obs, _ = env.reset()
    
    for step in range(3):  # Show first 3 steps
        print(f"\n{'='*80}")
        print(f"TIME STEP {step + 1}")
        print(f"{'='*80}\n")
        
        # Get actions
        actions = []
        for i, agent in enumerate(agents):
            action, _ = agent.get_action(obs[i])
            actions.append(action)
        
        actions = np.array(actions)
        
        # Print with clipping (as used in environment)
        task_id = 0
        for server_id in range(NUM_EDGE_SERVERS):
            print(f"Edge Server {server_id}:")
            server_action = actions[server_id].reshape(NUM_DEVICES_PER_SERVER, 2)
            
            for device_id in range(NUM_DEVICES_PER_SERVER):
                # Clip as done in environment
                alpha = np.clip(server_action[device_id, 0], 0, 1)
                mu = np.clip(server_action[device_id, 1], 0, 1)
                
                print(f"  Task {task_id:2d} (Device {device_id}): α={alpha:.4f}, μ={mu:.4f}  →  Local={alpha:.2%}, Edge={(1-alpha)*(1-mu):.2%}, Neighbors={(1-alpha)*mu:.2%}")
                task_id += 1
            print()
        
        # Step environment
        next_obs, rewards, terminated, truncated, info = env.step(actions)
        obs = next_obs
        
        if terminated or truncated:
            break
    
    print("="*80)
    print("\nLegend:")
    print("  α (alpha): Fraction executed locally on device")
    print("  μ (mu):    Fraction of offloaded task forwarded to neighbors")
    print("  Local:     α × task")
    print("  Edge:      (1-α) × (1-μ) × task")
    print("  Neighbors: (1-α) × μ × task")
    print("="*80)

if __name__ == "__main__":
    print_clipped_actions()
