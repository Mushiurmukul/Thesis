
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from mec_system.config import *
from mec_system.env.entities import MobileDevice, EdgeServer

class MECEnv(gym.Env):
    def __init__(self):
        super(MECEnv, self).__init__()
        
        self.num_servers = NUM_EDGE_SERVERS
        self.num_devices = NUM_DEVICES_PER_SERVER * NUM_EDGE_SERVERS
        
        # Initialize Entities
        self.servers = [EdgeServer(i) for i in range(self.num_servers)]
        self.devices = []
        for i in range(self.num_servers):
            for j in range(NUM_DEVICES_PER_SERVER):
                dev_id = i * NUM_DEVICES_PER_SERVER + j
                self.devices.append(MobileDevice(dev_id, associated_edge_id=i))
        
        # Define Action Space
        # For each device: [alpha_local, alpha_offload, mu_neighbor_1, mu_neighbor_2, ...]
        # Note: alpha_local + alpha_offload = 1. We can just output alpha_local.
        # And mu_neighbors sum <= 1.
        # To simplify for the agent, we can output raw logits and softmax them or normalize them.
        # Let's say action is a vector of size (Num_Devices, 1 + Num_Neighbors).
        # Actually, the paper says alpha_i_i and alpha_i_j. 
        # Let's output: [alpha_local, mu_neighbor_1, mu_neighbor_2, ... mu_neighbor_N-1]
        # Then alpha_offload = 1 - alpha_local.
        # And the portion kept at serving edge = alpha_offload * (1 - sum(mu_neighbors)).
        # Wait, the paper says: fraction mu_{i,j->k} is forwarded. 
        # "remaining portion (1-alpha) is offloaded... a fraction mu is forwarded... while rest remains at node j"
        # So: Local = alpha
        # Offloaded = 1 - alpha
        #   -> Forwarded to k = (1-alpha) * mu_k
        #   -> Kept at j = (1-alpha) * (1 - sum(mu_k))
        
        # Action dimension per device: 1 (alpha) + (Num_Servers - 1) (mu for each neighbor)
        self.action_dim_per_device = 1 + (self.num_servers - 1)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.num_devices, self.action_dim_per_device), dtype=np.float32)
        
        # Define Observation Space
        # Per server: Queue Length, Arrival Rate, Service Rate, Resource Loads (CPU, BW, Storage), Ambient Temp
        # Per device/task: Data Size, CPU Cycles, Deadline, Vulnerability
        # Global: DRF, Fairness
        # Let's flatten everything for simplicity or use Dict space.
        # Using Box for compatibility with standard MAPPO implementations usually.
        
        # Server State Dim: 7 (Queue, Lambda, Mu, Load_CPU, Load_BW, Load_ST, Temp)
        self.server_state_dim = 7
        # Task State Dim: 4 (Size, Cycles, Deadline, Vuln)
        self.task_state_dim = 4
        # Comm State Dim: 3 * Num_Neighbors (Uplink, Downlink, Inter-edge) - simplified to just current rates
        self.comm_state_dim = 1 + 1 + (self.num_servers - 1) # r_ij, r_ji, r_jk's
        
        # New State Metrics: DRF (1) + Task Reliability (Num_Devices_Per_Server)
        self.global_state_dim = 1 + NUM_DEVICES_PER_SERVER 
        
        self.obs_dim_per_agent = self.server_state_dim + NUM_DEVICES_PER_SERVER * self.task_state_dim + self.comm_state_dim + self.global_state_dim
        self.obs_space_dim = self.obs_dim_per_agent # Alias for compatibility
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_servers, self.obs_dim_per_agent), dtype=np.float32)

        # Dual Variables
        self.lambda_f = 0.0
        self.lambda_r = np.zeros(self.num_devices)
        self.lambda_d = np.zeros(self.num_devices)
        
        self.current_step = 0
        
        # Store last step metrics for observation
        self.last_drf = 1.0
        self.last_reliability = np.ones(self.num_devices)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        # Reset Entities
        for server in self.servers:
            server.queue = []
            server.update_load()
            
        # Reset Dual Variables
        self.lambda_f = 0.0
        self.lambda_r = np.zeros(self.num_devices)
        self.lambda_d = np.zeros(self.num_devices)
        
        self.last_drf = 1.0
        self.last_reliability = np.ones(self.num_devices)
        
        return self._get_obs(), {}

    def _get_obs(self):
        # Construct observation for each server (agent)
        obs = []
        for s_idx, server in enumerate(self.servers):
            server_obs = [
                server.get_queue_length(),
                server.arrival_rate,
                server.service_rate,
                server.current_cpu_load,
                server.current_bw_load,
                server.current_storage_load,
                25.0 # Ambient Temp (constant for now)
            ]
            
            # Task info for connected devices
            task_obs = []
            if hasattr(self, 'current_tasks'):
                my_tasks = self.current_tasks[s_idx * NUM_DEVICES_PER_SERVER : (s_idx + 1) * NUM_DEVICES_PER_SERVER]
                for task in my_tasks:
                    task_obs.extend([task.data_size, task.cpu_cycles_per_bit, task.deadline, task.vulnerability])
            else:
                task_obs = [0.0] * (NUM_DEVICES_PER_SERVER * 4)

            # Comm info
            comm_obs = [UPLINK_RATE_MAX, DOWNLINK_RATE_MAX] + [INTER_EDGE_RATE_MAX] * (self.num_servers - 1)
            
            # Global/Constraint Info
            # DRF (1)
            # Reliability (Num_Devices_Per_Server) for THIS agent's devices
            my_reliability = self.last_reliability[s_idx * NUM_DEVICES_PER_SERVER : (s_idx + 1) * NUM_DEVICES_PER_SERVER]
            global_obs = [self.last_drf] + my_reliability.tolist()
            
            obs.append(np.array(server_obs + task_obs + comm_obs + global_obs, dtype=np.float32))
            
        return np.array(obs)

    def step(self, actions):
        # actions: (Num_Servers, Num_Devices_Per_Server, Action_Dim)
        
        # 1. Generate new tasks for this step
        self.current_tasks = []
        for dev in self.devices:
            self.current_tasks.append(dev.generate_task(self.current_step))
            
        rewards = []
        
        # Metrics for plotting
        step_latency = []
        step_reliability = []
        step_success = 0
        step_total_tasks = 0
        
        # Global Fairness tracking
        # resource_loads: (Num_Servers, 3) -> CPU, BW, ST
        resource_loads = np.zeros((self.num_servers, 3)) 
        
        # --- Calculate Arrival Rates for Rho ---
        # We need to sum up arrivals from all sources to calculate rho for the M/M/1 formula
        # But we are inside the loop over devices.
        # We should calculate loads FIRST, then calculate delays?
        # Or we use the load from the *previous* step? 
        # The thesis implies steady state for the slot.
        # So we should calculate the total load on each server given the actions, 
        # THEN calculate the delays for each task.
        
        # Let's do a two-pass approach.
        # Pass 1: Calculate loads (arrival rates) on all servers.
        # Pass 2: Calculate delays and rewards.

        # --- Pass 1: Calculate Server Loads (Arrival Rates) ---
        server_arrival_rates = np.zeros(self.num_servers)
        
        for s_idx, server_action in enumerate(actions):
            my_tasks = self.current_tasks[s_idx * NUM_DEVICES_PER_SERVER : (s_idx + 1) * NUM_DEVICES_PER_SERVER]
            act_dim = 1 + (self.num_servers - 1)
            server_action = server_action.reshape(NUM_DEVICES_PER_SERVER, act_dim)
            
            for d_idx, task in enumerate(my_tasks):
                raw_alpha = server_action[d_idx, 0]
                alpha = np.clip(raw_alpha, 0, 1)
                raw_mus = server_action[d_idx, 1:]
                # Clip mus
                raw_mus = np.clip(raw_mus, 0, 1)
                
                if np.sum(raw_mus) > 1: mus = raw_mus / np.sum(raw_mus)
                else: mus = raw_mus
                
                # Local execution (no arrival at edge)
                
                # Offload to Serving Edge
                portion_edge = (1 - alpha) * (1 - np.sum(mus))
                if portion_edge > 0:
                    server_arrival_rates[s_idx] += portion_edge # Assuming 1 task = 1 unit, or weighted by size?
                    # M/M/1 usually uses lambda in jobs/second.
                    # Here we have portions of jobs.
                    
                # Offload to Neighbors
                neighbor_indices = [k for k in range(self.num_servers) if k != s_idx]
                for n_i, n_idx in enumerate(neighbor_indices):
                    portion_neighbor = (1 - alpha) * mus[n_i]
                    if portion_neighbor > 0:
                        server_arrival_rates[n_idx] += portion_neighbor

        # Update Server Rho
        # rho = lambda / mu
        # lambda = arrivals per second?
        # We have arrivals per time slot.
        # lambda (rate) = arrivals / TIME_SLOT_DURATION
        for s_idx in range(self.num_servers):
            arrival_rate = server_arrival_rates[s_idx] / TIME_SLOT_DURATION
            self.servers[s_idx].arrival_rate = arrival_rate
            # rho = lambda / mu
            self.servers[s_idx].current_cpu_load = arrival_rate / self.servers[s_idx].service_rate

        # --- Pass 2: Calculate Delays and Rewards ---
        current_step_reliability = np.zeros(self.num_devices)
        
        for s_idx, server_action in enumerate(actions):
            agent_reward = 0.0
            my_devices = self.devices[s_idx * NUM_DEVICES_PER_SERVER : (s_idx + 1) * NUM_DEVICES_PER_SERVER]
            my_tasks = self.current_tasks[s_idx * NUM_DEVICES_PER_SERVER : (s_idx + 1) * NUM_DEVICES_PER_SERVER]
            act_dim = 1 + (self.num_servers - 1)
            server_action = server_action.reshape(NUM_DEVICES_PER_SERVER, act_dim)
            
            for d_idx, task in enumerate(my_tasks):
                step_total_tasks += 1
                
                raw_mus = server_action[d_idx, 1:]
                # Clip mus to be non-negative!
                raw_mus = np.clip(raw_mus, 0, 1)
                
                if np.sum(raw_mus) > 1: mus = raw_mus / np.sum(raw_mus)
                else: mus = raw_mus
                
                raw_alpha = server_action[d_idx, 0]
                alpha = np.clip(raw_alpha, 0, 1)
                
                # --- Latency Calculation ---
                # 1. Local
                t_local = (alpha * task.total_cpu_cycles) / DEVICE_CPU_CAPACITY
                
                # 2. Edge (Serving)
                portion_edge = (1 - alpha) * (1 - np.sum(mus))
                t_tx_edge = (portion_edge * task.data_size) / UPLINK_RATE_MAX 
                t_proc_edge = (portion_edge * task.total_cpu_cycles) / EDGE_CPU_CAPACITY
                
                # Queue delay (M/M/1/K)
                rho = self.servers[s_idx].current_cpu_load
                if rho >= 0.99: rho = 0.99
                
                # M/M/1/K Formula
                K = MAX_QUEUE_SIZE
                p0 = (1 - rho) / (1 - rho**(K+1))
                pk = p0 * (rho**K)
                lq = (rho * (1 - (K+1)*rho**K + K*rho**(K+1))) / ((1 - rho) * (1 - rho**(K+1)))
                
                # W = (LQ - (1-P0)) / (lambda * (1-Pk))
                # Note: If lambda is 0, W is 0.
                if self.servers[s_idx].arrival_rate > 1e-9:
                    w_queue_edge = (lq - (1 - p0)) / (self.servers[s_idx].arrival_rate * (1 - pk))
                    # Ensure non-negative (numerical issues)
                    w_queue_edge = max(0.0, w_queue_edge)
                else:
                    w_queue_edge = 0.0
                
                t_edge = t_tx_edge + t_proc_edge + w_queue_edge
                
                # 3. Neighbor Edges
                t_neighbors = []
                neighbor_indices = [k for k in range(self.num_servers) if k != s_idx]
                for n_i, n_idx in enumerate(neighbor_indices):
                    portion_neighbor = (1 - alpha) * mus[n_i]
                    if portion_neighbor > 0:
                        t_tx_neighbor = (portion_neighbor * task.data_size) / INTER_EDGE_RATE_MAX
                        t_proc_neighbor = (portion_neighbor * task.total_cpu_cycles) / EDGE_CPU_CAPACITY 
                        
                        # Neighbor Queue
                        rho_n = self.servers[n_idx].current_cpu_load
                        if rho_n >= 0.99: rho_n = 0.99
                        
                        # M/M/1/K Formula for Neighbor
                        p0_n = (1 - rho_n) / (1 - rho_n**(K+1))
                        pk_n = p0_n * (rho_n**K)
                        lq_n = (rho_n * (1 - (K+1)*rho_n**K + K*rho_n**(K+1))) / ((1 - rho_n) * (1 - rho_n**(K+1)))
                        
                        if self.servers[n_idx].arrival_rate > 1e-9:
                            w_queue_neighbor = (lq_n - (1 - p0_n)) / (self.servers[n_idx].arrival_rate * (1 - pk_n))
                            w_queue_neighbor = max(0.0, w_queue_neighbor)
                        else:
                            w_queue_neighbor = 0.0
                        
                        t_neighbor = t_tx_edge + t_tx_neighbor + t_proc_neighbor + w_queue_neighbor
                        t_neighbors.append(t_neighbor)
                        
                        # Update Neighbor Load
                        resource_loads[n_idx, 0] += portion_neighbor * task.total_cpu_cycles / EDGE_CPU_CAPACITY
                        resource_loads[n_idx, 1] += portion_neighbor * task.data_size / EDGE_BANDWIDTH_CAPACITY
                    else:
                        t_neighbors.append(0)
                        
                # Total Latency
                T_total = max(t_local, t_edge, max(t_neighbors) if t_neighbors else 0)
                step_latency.append(T_total)
                
                # --- Reliability Calculation ---
                # R_trans = exp(-gamma * T_trans)
                # R_exe = exp(-lambda * nu * E_i)
                # R_queue = (1 - P_drop) * exp(-eta * L * W)
                
                # Local Reliability
                R_exe_local = np.exp(-PROCESSOR_FAULT_RATE_MAX * task.vulnerability * t_local)
                R_local = R_exe_local
                
                # Edge Reliability
                R_trans_edge = np.exp(-CHANNEL_ERROR_RATE_MAX * t_tx_edge)
                R_exe_edge = np.exp(-PROCESSOR_FAULT_RATE_MAX * task.vulnerability * t_proc_edge)
                
                # Queue Drop Prob
                p_drop_edge = pk # Already calculated Pk
                R_queue_edge = (1 - p_drop_edge) * np.exp(-QUEUE_SENSITIVITY * self.servers[s_idx].get_queue_length() * w_queue_edge)
                
                R_edge = R_queue_edge * R_trans_edge * R_exe_edge
                
                # Neighbor Reliability (Average of neighbors for simplicity in scalar metric, or min?)
                # Thesis says: "if executed at neighbor node k". 
                # Since we split the task, the total reliability is the product of reliabilities of all parts?
                # Or is it defined per task? The thesis defines R_i for "if executed at...".
                # But we have partitioning. 
                # Usually in partitioning, success means ALL parts succeed.
                # So R_total = R_local * R_edge * Product(R_neighbors)
                
                R_neighbors = 1.0
                for n_i, n_idx in enumerate(neighbor_indices):
                    portion_neighbor = (1 - alpha) * mus[n_i]
                    if portion_neighbor > 0:
                        t_n = t_neighbors[n_i]
                        # Re-calculate components for neighbor
                        t_tx_n = (portion_neighbor * task.data_size) / INTER_EDGE_RATE_MAX
                        t_proc_n = (portion_neighbor * task.total_cpu_cycles) / EDGE_CPU_CAPACITY
                        
                        # Use neighbor queue stats
                        rho_n = self.servers[n_idx].current_cpu_load
                        if rho_n >= 0.99: rho_n = 0.99
                        
                        # M/M/1/K stats for neighbor
                        p0_n = (1 - rho_n) / (1 - rho_n**(K+1))
                        pk_n = p0_n * (rho_n**K)
                        lq_n = (rho_n * (1 - (K+1)*rho_n**K + K*rho_n**(K+1))) / ((1 - rho_n) * (1 - rho_n**(K+1)))
                        
                        if self.servers[n_idx].arrival_rate > 1e-9:
                            w_queue_n = (lq_n - (1 - p0_n)) / (self.servers[n_idx].arrival_rate * (1 - pk_n))
                            w_queue_n = max(0.0, w_queue_n)
                        else:
                            w_queue_n = 0.0
                        
                        R_trans_n = np.exp(-CHANNEL_ERROR_RATE_MAX * (t_tx_edge + t_tx_n)) # Path i->j->k
                        R_exe_n = np.exp(-PROCESSOR_FAULT_RATE_MAX * task.vulnerability * t_proc_n)
                        
                        p_drop_n = pk_n
                        R_queue_n = (1 - p_drop_n) * np.exp(-QUEUE_SENSITIVITY * self.servers[n_idx].get_queue_length() * w_queue_n)
                        
                        R_neighbors *= (R_queue_n * R_trans_n * R_exe_n)
                
                # Composite Reliability
                R_total = 1.0
                if alpha > 0: R_total *= R_local
                if portion_edge > 0: R_total *= R_edge
                R_total *= R_neighbors
                
                step_reliability.append(R_total)
                
                global_idx = s_idx * NUM_DEVICES_PER_SERVER + d_idx
                current_step_reliability[global_idx] = R_total
                
                # Update Server Load (Serving Edge)
                resource_loads[s_idx, 0] += portion_edge * task.total_cpu_cycles / EDGE_CPU_CAPACITY
                resource_loads[s_idx, 1] += portion_edge * task.data_size / EDGE_BANDWIDTH_CAPACITY
                
                # --- Reward Calculation ---
                T_hat = T_total / task.deadline
                if T_hat <= 1: step_success += 1
                
                # Primary Objective
                obj = BETA * R_total - (1 - BETA) * T_hat
                
                # Constraints
                viol_r = max(0, RELIABILITY_THRESHOLD - R_total)
                viol_d = 1.0 if T_hat > 1 else 0.0
                
                # Update Duals
                self.lambda_r[global_idx] = max(0, self.lambda_r[global_idx] + LEARNING_RATE_DUAL * viol_r)
                self.lambda_d[global_idx] = max(0, self.lambda_d[global_idx] + LEARNING_RATE_DUAL * viol_d)
                
                penalty = self.lambda_r[global_idx] * viol_r + self.lambda_d[global_idx] * viol_d
                agent_reward += (obj - penalty)
            
            rewards.append(agent_reward)

        # --- Fairness (DRF) ---
        # s_j = max_r (L_j^r)
        dominant_shares = np.max(resource_loads, axis=1) # (Num_Servers,)
        
        # Jain's Index on Dominant Shares
        # F = (sum s_j)^2 / (N * sum s_j^2)
        sum_s = np.sum(dominant_shares)
        sum_sq_s = np.sum(dominant_shares**2)
        
        if sum_sq_s > 0:
            drf = (sum_s**2) / (self.num_servers * sum_sq_s)
        else:
            drf = 1.0
            
        viol_f = max(0, FAIRNESS_THRESHOLD - drf)
        self.lambda_f = max(0, self.lambda_f + LEARNING_RATE_DUAL * viol_f)
        
        # Apply global fairness penalty
        for i in range(self.num_servers):
            rewards[i] -= self.lambda_f * viol_f
            
        self.current_step += 1
        truncated = False
        terminated = self.current_step >= NUM_TIME_STEPS
        
        # Update state for next observation
        self.last_drf = drf
        self.last_reliability = current_step_reliability
        
        info = {
            "avg_latency": np.mean(step_latency) if step_latency else 0,
            "avg_reliability": np.mean(step_reliability) if step_reliability else 0,
            "success_rate": step_success / step_total_tasks if step_total_tasks > 0 else 0,
            "drf_fairness": drf,
            "lambda_f": self.lambda_f,
            "avg_lambda_r": np.mean(self.lambda_r),
            "avg_lambda_d": np.mean(self.lambda_d)
        }
        
        return self._get_obs(), np.array(rewards), terminated, truncated, info

