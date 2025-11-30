
import time
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mec_system.env.mec_env import MECEnv
from mec_system.agents.mappo import MAPPOAgent
from mec_system.config import *

def estimate_time():
    print("Estimating training time for 1000 episodes...")
    
    # Setup
    env = MECEnv()
    agents = []
    for i in range(NUM_EDGE_SERVERS):
        act_dim = NUM_DEVICES_PER_SERVER * (1 + (NUM_EDGE_SERVERS - 1))
        agents.append(MAPPOAgent(env.obs_space_dim, act_dim))
        
    # Warmup (1 episode)
    print("Warming up...")
    obs, _ = env.reset()
    buffers = [[] for _ in range(NUM_EDGE_SERVERS)]
    for step in range(NUM_TIME_STEPS):
        actions = []
        action_log_probs = []
        for i, agent in enumerate(agents):
            action, log_prob = agent.get_action(obs[i])
            actions.append(action)
            action_log_probs.append(log_prob)
        actions = np.array(actions)
        next_obs, rewards, terminated, truncated, info = env.step(actions)
        done = terminated or truncated
        for i in range(NUM_EDGE_SERVERS):
            buffers[i].append((obs[i], actions[i], rewards[i], next_obs[i], action_log_probs[i], done))
        obs = next_obs
        if done: break
    for i, agent in enumerate(agents):
        agent.update(buffers[i])
        
    # Measurement (5 episodes)
    num_test_episodes = 5
    print(f"Measuring time for {num_test_episodes} episodes...")
    start_time = time.time()
    
    for episode in range(num_test_episodes):
        obs, _ = env.reset()
        buffers = [[] for _ in range(NUM_EDGE_SERVERS)]
        
        for step in range(NUM_TIME_STEPS):
            actions = []
            action_log_probs = []
            for i, agent in enumerate(agents):
                action, log_prob = agent.get_action(obs[i])
                actions.append(action)
                action_log_probs.append(log_prob)
            actions = np.array(actions)
            next_obs, rewards, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            for i in range(NUM_EDGE_SERVERS):
                buffers[i].append((obs[i], actions[i], rewards[i], next_obs[i], action_log_probs[i], done))
            obs = next_obs
            if done: break
            
        for i, agent in enumerate(agents):
            agent.update(buffers[i])
            
    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_ep = total_time / num_test_episodes
    
    est_1000 = avg_time_per_ep * 1000
    
    print(f"Time for {num_test_episodes} episodes: {total_time:.2f}s")
    print(f"Average time per episode: {avg_time_per_ep:.2f}s")
    print(f"Estimated time for 1000 episodes: {est_1000:.2f}s ({est_1000/60:.2f} minutes)")

if __name__ == "__main__":
    estimate_time()
