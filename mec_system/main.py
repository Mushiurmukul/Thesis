
import numpy as np
import torch
from mec_system.env.mec_env import MECEnv
from mec_system.agents.mappo import MAPPOAgent
from mec_system.config import *
import matplotlib.pyplot as plt

def main():
    env = MECEnv()
    
    # Initialize Agents
    # Assuming homogeneous agents for now, sharing weights or individual?
    # Thesis implies "Each agent j... uses its learned policy".
    # We can use independent learners or shared policy.
    # Let's use Independent Learners (IL) for now as it's simpler to map to "Agent j".
    
    agents = []
    for i in range(NUM_EDGE_SERVERS):
        # Action dim per server = Num_Devices * (1 + Neighbors)
        act_dim = NUM_DEVICES_PER_SERVER * (1 + (NUM_EDGE_SERVERS - 1))
        agents.append(MAPPOAgent(env.obs_space_dim, act_dim))
        
    num_episodes = 3000 # Extended for ultra-smooth convergence
    
    # Metrics History
    history = {
        "reward": [],
        "latency": [],
        "reliability": [],
        "success_rate": [],
        "fairness": [],
        "lambda_f": [],
        "lambda_r": [],
        "lambda_d": []
    }
    
    print(f"{'Episode':<8} | {'Reward':<10} | {'Latency':<8} | {'Reliab':<8} | {'Success':<8} | {'Fairness':<8} | {'L_Fair':<8} | {'L_Rel':<8} | {'L_Delay':<8}")
    print("-" * 100)
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        
        # Episode accumulators
        ep_latency = []
        ep_reliability = []
        ep_success = []
        ep_fairness = []
        ep_lambda_f = []
        ep_lambda_r = []
        ep_lambda_d = []
        
        # Buffers for each agent
        buffers = [[] for _ in range(NUM_EDGE_SERVERS)]
        
        for step in range(NUM_TIME_STEPS):
            # 1. Select Actions
            actions = []
            action_log_probs = []
            
            for i, agent in enumerate(agents):
                action, log_prob = agent.get_action(obs[i])
                actions.append(action)
                action_log_probs.append(log_prob)
                
            actions = np.array(actions)
            
            # 2. Step Environment
            next_obs, rewards, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            
            # Store metrics
            ep_latency.append(info["avg_latency"])
            ep_reliability.append(info["avg_reliability"])
            ep_success.append(info["success_rate"])
            ep_fairness.append(info["drf_fairness"])
            ep_lambda_f.append(info["lambda_f"])
            ep_lambda_r.append(info["avg_lambda_r"])
            ep_lambda_d.append(info["avg_lambda_d"])
            
            # 3. Store in Buffer
            for i in range(NUM_EDGE_SERVERS):
                buffers[i].append((obs[i], actions[i], rewards[i], next_obs[i], action_log_probs[i], done))
                
            obs = next_obs
            episode_reward += np.sum(rewards)
            
            if done:
                break
                
        # 4. Update Agents
        loss_a = 0
        loss_c = 0
        for i, agent in enumerate(agents):
            l_a, l_c = agent.update(buffers[i])
            loss_a += l_a
            loss_c += l_c
            
        # Store Episode Metrics
        history["reward"].append(episode_reward)
        history["latency"].append(np.mean(ep_latency))
        history["reliability"].append(np.mean(ep_reliability))
        history["success_rate"].append(np.mean(ep_success))
        history["fairness"].append(np.mean(ep_fairness))
        history["lambda_f"].append(np.mean(ep_lambda_f))
        history["lambda_r"].append(np.mean(ep_lambda_r))
        history["lambda_d"].append(np.mean(ep_lambda_d))
        
        print(f"{episode+1:<8} | {episode_reward:<10.2f} | {np.mean(ep_latency):<8.4f} | {np.mean(ep_reliability):<8.4f} | {np.mean(ep_success):<8.4f} | {np.mean(ep_fairness):<8.4f} | {np.mean(ep_lambda_f):<8.4f} | {np.mean(ep_lambda_r):<8.4f} | {np.mean(ep_lambda_d):<8.4f}")
        
    print("Training Complete.")
    
    # Plotting (6-panel grid)
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Reward
    axs[0, 0].plot(history["reward"])
    axs[0, 0].set_title("Training Reward")
    axs[0, 0].set_xlabel("Episode")
    axs[0, 0].set_ylabel("Reward")
    axs[0, 0].grid(True, alpha=0.3)
    
    # 2. Latency
    axs[0, 1].plot(history["latency"], color='orange')
    axs[0, 1].set_title("Average Latency")
    axs[0, 1].set_xlabel("Episode")
    axs[0, 1].set_ylabel("Avg Latency (s)")
    axs[0, 1].grid(True, alpha=0.3)
    
    # 3. Reliability
    axs[0, 2].plot(history["reliability"], color='green')
    axs[0, 2].set_title("Average Reliability")
    axs[0, 2].set_xlabel("Episode")
    axs[0, 2].set_ylabel("Avg Reliability")
    axs[0, 2].grid(True, alpha=0.3)
    
    # 4. Success Rate
    axs[1, 0].plot(history["success_rate"], color='red')
    axs[1, 0].set_title("Task Success Rate")
    axs[1, 0].set_xlabel("Episode")
    axs[1, 0].set_ylabel("Success Rate")
    axs[1, 0].grid(True, alpha=0.3)
    
    # 5. Fairness
    axs[1, 1].plot(history["fairness"], color='purple')
    axs[1, 1].set_title("System Fairness (DRF)")
    axs[1, 1].set_xlabel("Episode")
    axs[1, 1].set_ylabel("DRF Fairness")
    axs[1, 1].grid(True, alpha=0.3)
    
    # 6. Lagrange Multipliers
    axs[1, 2].plot(history["lambda_f"], label='Lambda Fairness', color='purple', linestyle='--')
    axs[1, 2].plot(history["lambda_r"], label='Lambda Reliability', color='green', linestyle='--')
    axs[1, 2].plot(history["lambda_d"], label='Lambda Delay', color='orange', linestyle='--')
    axs[1, 2].set_title("Lagrange Multipliers")
    axs[1, 2].set_xlabel("Episode")
    axs[1, 2].set_ylabel("Multiplier Value")
    axs[1, 2].legend()
    axs[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("training_results_grid.png")
    print("Saved training_results_grid.png")

if __name__ == "__main__":
    main()
