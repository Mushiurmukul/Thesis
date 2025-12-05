
import numpy as np
import torch
from mec_system.env.mec_env import MECEnv
from mec_system.agents.mappo import MAPPOAgent
from mec_system.config import Config
import matplotlib.pyplot as plt

def main():
    env = MECEnv()
    
    # Initialize Agents
    agents = []
    for i in range(Config.system.NUM_EDGE_SERVERS):
        act_dim = Config.system.NUM_DEVICES_PER_SERVER * 2
        agents.append(MAPPOAgent(env.obs_space_dim, act_dim))
        
    print(f"Action Space per Agent: {act_dim} (Flattened)")
    print(f"  - Devices per Server: {Config.system.NUM_DEVICES_PER_SERVER}")
    print(f"  - Action Components per Device: 2 (alpha, mu)")
    print("-" * 50)
        
    num_episodes = 12000
    
    # Metrics History
    history = {
        "reward": [],
        "latency": [],
        "reliability": [],
        "success_rate": [],
        "fairness": [],
        "lambda_f": [],
        "lambda_r": [],
        "lambda_d": [],
        "actor_loss": [],
        "critic_loss": []
    }
    
    # Open file for action logging
    with open("action.txt", "w") as f, open("task.txt", "w") as f_task:
        f.write("Episode,Step,Server,Device,Alpha,Mu,Task_Size,Local_Bits,Edge_Bits,Neighbor_Bits\n")
        f_task.write("Episode,Step,Server,Device,Task_ID,Data_Size,CPU_Cycles,Deadline,Total_Latency\n")
        
        print(f"{'Episode':<8} | {'Reward':<10} | {'Latency':<8} | {'Reliab':<8} | {'Success':<8} | {'Fairness':<8} | {'Lam_F':<8} | {'Lam_R':<8} | {'Lam_D':<8}")
        print("-" * 100)
        
        for episode in range(num_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            
            ep_latency = []
            ep_reliability = []
            ep_success = []
            ep_fairness = []
            ep_lambda_f = []
            ep_lambda_r = []
            ep_lambda_d = []
            
            buffers = [[] for _ in range(Config.system.NUM_EDGE_SERVERS)]
            
            for step in range(Config.system.NUM_TIME_STEPS):
                # 1. Select Actions
                actions = []
                action_log_probs = []
                
                for i, agent in enumerate(agents):
                    action, log_prob = agent.get_action(obs[i])
                    actions.append(action)
                    action_log_probs.append(log_prob)
                    
                actions = np.array(actions)
                
                # --- LOGGING ACTIONS ---
                for s_idx in range(Config.system.NUM_EDGE_SERVERS):
                    server_action = actions[s_idx].reshape(Config.system.NUM_DEVICES_PER_SERVER, 2)
                    start_task_idx = s_idx * Config.system.NUM_DEVICES_PER_SERVER
                    
                    for d_idx in range(Config.system.NUM_DEVICES_PER_SERVER):
                        alpha = np.clip(server_action[d_idx, 0], 0, 1)
                        mu = np.clip(server_action[d_idx, 1], 0, 1)
                        
                        task = env.current_tasks[start_task_idx + d_idx]
                        task_size = task.data_size
                        
                        local_bits = task_size * alpha
                        edge_bits = task_size * (1 - alpha) * (1 - mu)
                        neighbor_bits = task_size * (1 - alpha) * mu
                        
                        f.write(f"{episode+1},{step+1},{s_idx},{d_idx},{alpha:.6f},{mu:.6f},{task_size:.2f},{local_bits:.2f},{edge_bits:.2f},{neighbor_bits:.2f}\n")
                
                # Capture tasks before step (as step generates new tasks)
                tasks_this_step = env.current_tasks
                
                # 2. Step Environment
                next_obs, rewards, terminated, truncated, info = env.step(actions)
                
                # --- LOGGING TASKS WITH LATENCY ---
                latencies = info.get("latencies", [])
                for s_idx in range(Config.system.NUM_EDGE_SERVERS):
                    start_task_idx = s_idx * Config.system.NUM_DEVICES_PER_SERVER
                    for d_idx in range(Config.system.NUM_DEVICES_PER_SERVER):
                        global_idx = start_task_idx + d_idx
                        task = tasks_this_step[global_idx]
                        latency = latencies[global_idx] if global_idx < len(latencies) else 0.0
                        
                        f_task.write(f"{episode+1},{step+1},{s_idx},{d_idx},{task.task_id},{task.data_size:.2f},{task.total_cpu_cycles:.2f},{task.deadline:.4f},{latency:.4f}\n")
                done = terminated or truncated
                
                ep_latency.append(info["avg_latency"])
                ep_reliability.append(info["avg_reliability"])
                ep_success.append(info["success_rate"])
                ep_fairness.append(info["drf_fairness"])
                ep_lambda_f.append(info["lambda_f"])
                ep_lambda_r.append(info["avg_lambda_r"])
                ep_lambda_d.append(info["avg_lambda_d"])
                
                # 3. Store in Buffer
                for i in range(Config.system.NUM_EDGE_SERVERS):
                    buffers[i].append((obs[i], actions[i], rewards[i], next_obs[i], action_log_probs[i], done))
                    
                obs = next_obs
                episode_reward += np.sum(rewards)
                
                if done:
                    break
                    
            # 4. Update Agents (Decentralized)
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
            history["actor_loss"].append(loss_a / Config.system.NUM_EDGE_SERVERS)
            history["critic_loss"].append(loss_c / Config.system.NUM_EDGE_SERVERS)
            
            print(f"{episode+1:<8} | {episode_reward:<10.2f} | {np.mean(ep_latency):<8.4f} | {np.mean(ep_reliability):<8.4f} | {np.mean(ep_success):<8.4f} | {np.mean(ep_fairness):<8.4f} | {np.mean(ep_lambda_f):<8.4f} | {np.mean(ep_lambda_r):<8.4f} | {np.mean(ep_lambda_d):<8.4f}")
            
    print("Training Complete.")
    
    # Plotting
    fig, axs = plt.subplots(2, 4, figsize=(24, 10))
    
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
    
    # 7. Actor Loss
    axs[1, 3].plot(history["actor_loss"], color='brown')
    axs[1, 3].set_title("Actor Loss")
    axs[1, 3].set_xlabel("Episode")
    axs[1, 3].set_ylabel("Loss")
    axs[1, 3].grid(True, alpha=0.3)
    
    # 8. Critic Loss
    axs[0, 3].plot(history["critic_loss"], color='teal')
    axs[0, 3].set_title("Critic Loss")
    axs[0, 3].set_xlabel("Episode")
    axs[0, 3].set_ylabel("Loss")
    axs[0, 3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("training_results_grid.png")
    print("Saved training_results_grid.png")

if __name__ == "__main__":
    main()
