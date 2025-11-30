
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from mec_system.config import Config
from mec_system.env.entities import MobileDevice, EdgeServer

class MECEnv(gym.Env):
    def __init__(self):
        super(MECEnv, self).__init__()
        
        self.num_servers = Config.system.NUM_EDGE_SERVERS
        self.num_devices = Config.system.NUM_DEVICES_PER_SERVER * Config.system.NUM_EDGE_SERVERS
        
        # Initialize Entities
        self.servers = [EdgeServer(i) for i in range(self.num_servers)]
        self.devices = []
        for i in range(self.num_servers):
            for j in range(Config.system.NUM_DEVICES_PER_SERVER):
                dev_id = i * Config.system.NUM_DEVICES_PER_SERVER + j
                self.devices.append(MobileDevice(dev_id, associated_edge_id=i))
        
        # Define Action Space
        self.action_dim_per_device = 2
        self.action_space = spaces.Box(low=0, high=1, shape=(self.num_devices, self.action_dim_per_device), dtype=np.float32)
        
        # Define Observation Space
        self.server_state_dim = 8  # Own server: 8 features
        self.task_state_dim = 4    # Per task: 4 features
        self.neighbor_state_dim = 4 * (self.num_servers - 1)  # Each neighbor: queue, queue_util, cpu_load, arrival_rate
        self.comm_state_dim = 1 + 1 + (self.num_servers - 1)
        
        self.global_state_dim = 1 + Config.system.NUM_DEVICES_PER_SERVER 
        
        self.obs_dim_per_agent = (self.server_state_dim + 
                                  Config.system.NUM_DEVICES_PER_SERVER * self.task_state_dim + 
                                  self.neighbor_state_dim +  # NEW: Neighbor observations
                                  self.comm_state_dim + 
                                  self.global_state_dim)
        self.obs_space_dim = self.obs_dim_per_agent
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_servers, self.obs_dim_per_agent), dtype=np.float32)

        # Dual Variables
        self.lambda_f = 0.1
        self.lambda_r = np.ones(self.num_devices) * 0.1
        self.lambda_d = np.ones(self.num_devices) * 0.1
        
        self.current_step = 0
        self.last_drf = 1.0
        self.last_reliability = np.ones(self.num_devices)
        
    def _generate_tasks(self):
        self.current_tasks = []
        for dev in self.devices:
            self.current_tasks.append(dev.generate_task(self.current_step))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        for server in self.servers:
            server.queue = []
            server.update_load()
            
        self.last_drf = 1.0
        self.last_reliability = np.ones(self.num_devices)
        
        self._generate_tasks()
        
        return self._get_obs(), {}

    def _get_obs(self):
        obs = []
        for s_idx, server in enumerate(self.servers):
            # Calculate queue utilization (0-1 range) to make congestion more explicit
            queue_utilization = server.get_queue_length() / Config.system.MAX_QUEUE_SIZE
            
            server_obs = [
                server.get_queue_length(),
                queue_utilization,  # Normalized queue state (0-1)
                server.arrival_rate,
                server.service_rate,
                server.current_cpu_load,
                server.current_bw_load,
                server.current_storage_load,
                25.0
            ]
            # Task observations (Normalized)
            task_obs = []
            my_tasks = self.current_tasks[s_idx * Config.system.NUM_DEVICES_PER_SERVER : (s_idx + 1) * Config.system.NUM_DEVICES_PER_SERVER]
            for task in my_tasks:
                # Normalize task features
                norm_data_size = task.data_size / Config.task.DATA_SIZE_MAX
                norm_cycles = task.cpu_cycles_per_bit / Config.task.CPU_CYCLES_PER_BIT_MAX
                norm_deadline = task.deadline / Config.task.DEADLINE_MAX
                task_obs.extend([norm_data_size, norm_cycles, norm_deadline, task.vulnerability])

            # NEW: Neighbor server states
            neighbor_obs = []
            for n_idx in range(self.num_servers):
                if n_idx != s_idx:
                    neighbor = self.servers[n_idx]
                    neighbor_queue_util = neighbor.get_queue_length() / Config.system.MAX_QUEUE_SIZE
                    neighbor_obs.extend([
                        neighbor.get_queue_length() / Config.system.MAX_QUEUE_SIZE, # Normalize queue length too
                        neighbor_queue_util,
                        neighbor.current_cpu_load,
                        neighbor.arrival_rate / Config.system.NUM_DEVICES_PER_SERVER # Normalize arrival rate
                    ])

            # Normalize communication rates
            comm_obs = [
                Config.network.UPLINK_RATE_MAX / Config.network.UPLINK_RATE_MAX, # 1.0
                Config.network.DOWNLINK_RATE_MAX / Config.network.DOWNLINK_RATE_MAX, # 1.0
                # Inter-edge rate is usually high, normalize by itself or max possible
            ] + [Config.network.INTER_EDGE_RATE_MAX / Config.network.INTER_EDGE_RATE_MAX] * (self.num_servers - 1)
            
            my_reliability = self.last_reliability[s_idx * Config.system.NUM_DEVICES_PER_SERVER : (s_idx + 1) * Config.system.NUM_DEVICES_PER_SERVER]
            global_obs = [self.last_drf] + my_reliability.tolist()
            
            obs.append(np.array(server_obs + task_obs + neighbor_obs + comm_obs + global_obs, dtype=np.float32))
            
        return np.array(obs)
    
    def _get_global_state(self):
        """Get global state for centralized critic (sees all servers)"""
        global_state = []
        
        # Add all servers' states
        for server in self.servers:
            queue_utilization = server.get_queue_length() / Config.system.MAX_QUEUE_SIZE
            global_state.extend([
                server.get_queue_length(),
                queue_utilization,
                server.arrival_rate,
                server.service_rate,
                server.current_cpu_load,
                server.current_bw_load,
                server.current_storage_load
            ])
        
        # Add global metrics
        global_state.append(self.last_drf)
        global_state.extend(self.last_reliability.tolist())
        
        return np.array(global_state, dtype=np.float32)

    def step(self, actions):
        rewards = []
        step_latency = []
        step_reliability = []
        step_success = 0
        step_total_tasks = 0
        
        resource_loads = np.zeros((self.num_servers, 3)) 
        server_arrival_rates = np.zeros(self.num_servers)
        
        for s_idx, server_action in enumerate(actions):
            my_tasks = self.current_tasks[s_idx * Config.system.NUM_DEVICES_PER_SERVER : (s_idx + 1) * Config.system.NUM_DEVICES_PER_SERVER]
            act_dim = 2
            server_action = server_action.reshape(Config.system.NUM_DEVICES_PER_SERVER, act_dim)
            
            for d_idx, task in enumerate(my_tasks):
                alpha = np.clip(server_action[d_idx, 0], 0, 1)
                mu = np.clip(server_action[d_idx, 1], 0, 1)
                
                portion_edge = (1 - alpha) * (1 - mu)
                if portion_edge > 0:
                    server_arrival_rates[s_idx] += portion_edge
                    
                portion_neighbors_total = (1 - alpha) * mu
                if portion_neighbors_total > 0:
                    neighbor_indices = [k for k in range(self.num_servers) if k != s_idx]
                    num_neighbors = len(neighbor_indices)
                    portion_per_neighbor = portion_neighbors_total / num_neighbors
                    
                    for n_idx in neighbor_indices:
                        server_arrival_rates[n_idx] += portion_per_neighbor

        for s_idx in range(self.num_servers):
            arrival_rate = server_arrival_rates[s_idx] / Config.system.TIME_SLOT_DURATION
            self.servers[s_idx].arrival_rate = arrival_rate
            self.servers[s_idx].current_cpu_load = arrival_rate / self.servers[s_idx].service_rate

        current_step_reliability = np.zeros(self.num_devices)
        
        for s_idx, server_action in enumerate(actions):
            agent_reward = 0.0
            my_tasks = self.current_tasks[s_idx * Config.system.NUM_DEVICES_PER_SERVER : (s_idx + 1) * Config.system.NUM_DEVICES_PER_SERVER]
            act_dim = 2
            server_action = server_action.reshape(Config.system.NUM_DEVICES_PER_SERVER, act_dim)
            
            for d_idx, task in enumerate(my_tasks):
                step_total_tasks += 1
                
                alpha = np.clip(server_action[d_idx, 0], 0, 1)
                mu = np.clip(server_action[d_idx, 1], 0, 1)
                
                # Sample dynamic transmission rates (realistic channel variation)
                uplink_rate = np.random.uniform(Config.network.UPLINK_RATE_MIN, Config.network.UPLINK_RATE_MAX)
                downlink_rate = np.random.uniform(Config.network.DOWNLINK_RATE_MIN, Config.network.DOWNLINK_RATE_MAX)
                inter_edge_rate = np.random.uniform(Config.network.INTER_EDGE_RATE_MIN, Config.network.INTER_EDGE_RATE_MAX)
                
                t_local = (alpha * task.total_cpu_cycles) / Config.system.DEVICE_CPU_CAPACITY
                
                portion_edge = (1 - alpha) * (1 - mu)
                t_tx_edge = (portion_edge * task.data_size) / uplink_rate  # Use sampled rate
                t_proc_edge = (portion_edge * task.total_cpu_cycles) / Config.system.EDGE_CPU_CAPACITY
                
                rho = self.servers[s_idx].current_cpu_load
                if rho >= 0.99: rho = 0.99
                
                K = Config.system.MAX_QUEUE_SIZE
                p0 = (1 - rho) / (1 - rho**(K+1))
                pk = p0 * (rho**K)
                lq = (rho * (1 - (K+1)*rho**K + K*rho**(K+1))) / ((1 - rho) * (1 - rho**(K+1)))
                
                if self.servers[s_idx].arrival_rate > 1e-9:
                    w_queue_edge = (lq - (1 - p0)) / (self.servers[s_idx].arrival_rate * (1 - pk))
                    w_queue_edge = max(0.0, w_queue_edge)
                else:
                    w_queue_edge = 0.0
                
                t_dl_edge = (portion_edge * task.result_size) / downlink_rate  # Use sampled rate
                t_edge = t_tx_edge + t_proc_edge + t_dl_edge + w_queue_edge
                
                t_neighbors = []
                neighbor_indices = [k for k in range(self.num_servers) if k != s_idx]
                num_neighbors = len(neighbor_indices)
                portion_neighbors_total = (1 - alpha) * mu
                portion_per_neighbor = portion_neighbors_total / num_neighbors if num_neighbors > 0 else 0
                
                for n_idx in neighbor_indices:
                    if portion_per_neighbor > 0:
                        t_tx_neighbor = (portion_per_neighbor * task.data_size) / inter_edge_rate  # Use sampled rate
                        t_proc_neighbor = (portion_per_neighbor * task.total_cpu_cycles) / Config.system.EDGE_CPU_CAPACITY 
                        
                        rho_n = self.servers[n_idx].current_cpu_load
                        if rho_n >= 0.99: rho_n = 0.99
                        
                        p0_n = (1 - rho_n) / (1 - rho_n**(K+1))
                        pk_n = p0_n * (rho_n**K)
                        lq_n = (rho_n * (1 - (K+1)*rho_n**K + K*rho_n**(K+1))) / ((1 - rho_n) * (1 - rho_n**(K+1)))
                        
                        if self.servers[n_idx].arrival_rate > 1e-9:
                            w_queue_neighbor = (lq_n - (1 - p0_n)) / (self.servers[n_idx].arrival_rate * (1 - pk_n))
                            w_queue_neighbor = max(0.0, w_queue_neighbor)
                        else:
                            w_queue_neighbor = 0.0
                        
                        t_dl_neighbor = (portion_per_neighbor * task.result_size) / downlink_rate  # Use sampled rate

                        t_neighbor = t_tx_edge + t_tx_neighbor + t_proc_neighbor + t_dl_neighbor + w_queue_neighbor
                        t_neighbors.append(t_neighbor)
                        
                        resource_loads[n_idx, 0] += portion_per_neighbor * task.total_cpu_cycles / Config.system.EDGE_CPU_CAPACITY
                        resource_loads[n_idx, 1] += portion_per_neighbor * task.data_size / Config.system.EDGE_BANDWIDTH_CAPACITY
                    else:
                        t_neighbors.append(0)
                        
                T_total = max(t_local, t_edge, max(t_neighbors) if t_neighbors else 0) + Config.task.RECOM_TIME
                step_latency.append(T_total)
                
                R_exe_local = np.exp(-Config.reliability.PROCESSOR_FAULT_RATE_MAX * task.vulnerability * t_local)
                R_local = R_exe_local
                
                R_trans_edge = np.exp(-Config.reliability.CHANNEL_ERROR_RATE_MAX * t_tx_edge)
                R_exe_edge = np.exp(-Config.reliability.PROCESSOR_FAULT_RATE_MAX * task.vulnerability * t_proc_edge)
                
                p_drop_edge = pk
                # FIX: Use lq (modeled queue length) instead of get_queue_length() (instantaneous)
                # This aligns reliability with the latency calculation
                R_queue_edge = (1 - p_drop_edge) * np.exp(-Config.reliability.QUEUE_SENSITIVITY * lq * w_queue_edge)
                
                R_edge = R_queue_edge * R_trans_edge * R_exe_edge
                
                R_neighbors = 1.0
                for n_i, n_idx in enumerate(neighbor_indices):
                    if portion_per_neighbor > 0:
                        t_tx_n = (portion_per_neighbor * task.data_size) / Config.network.INTER_EDGE_RATE_MAX
                        t_proc_n = (portion_per_neighbor * task.total_cpu_cycles) / Config.system.EDGE_CPU_CAPACITY
                        
                        rho_n = self.servers[n_idx].current_cpu_load
                        if rho_n >= 0.99: rho_n = 0.99
                        
                        p0_n = (1 - rho_n) / (1 - rho_n**(K+1))
                        pk_n = p0_n * (rho_n**K)
                        lq_n = (rho_n * (1 - (K+1)*rho_n**K + K*rho_n**(K+1))) / ((1 - rho_n) * (1 - rho_n**(K+1)))
                        
                        if self.servers[n_idx].arrival_rate > 1e-9:
                            w_queue_n = (lq_n - (1 - p0_n)) / (self.servers[n_idx].arrival_rate * (1 - pk_n))
                            w_queue_n = max(0.0, w_queue_n)
                        else:
                            w_queue_n = 0.0
                        
                        R_trans_n = np.exp(-Config.reliability.CHANNEL_ERROR_RATE_MAX * (t_tx_edge + t_tx_n))
                        R_exe_n = np.exp(-Config.reliability.PROCESSOR_FAULT_RATE_MAX * task.vulnerability * t_proc_n)
                        
                        p_drop_n = pk_n
                        R_queue_n = (1 - p_drop_n) * np.exp(-Config.reliability.QUEUE_SENSITIVITY * self.servers[n_idx].get_queue_length() * w_queue_n)
                        
                        R_neighbors *= (R_queue_n * R_trans_n * R_exe_n)
                
                R_total = 1.0
                if alpha > 0: R_total *= R_local
                if portion_edge > 0: R_total *= R_edge
                R_total *= R_neighbors
                
                step_reliability.append(R_total)
                
                global_idx = s_idx * Config.system.NUM_DEVICES_PER_SERVER + d_idx
                current_step_reliability[global_idx] = R_total
                
                resource_loads[s_idx, 0] += portion_edge * task.total_cpu_cycles / Config.system.EDGE_CPU_CAPACITY
                resource_loads[s_idx, 1] += portion_edge * task.data_size / Config.system.EDGE_BANDWIDTH_CAPACITY
                
                T_hat = T_total / task.deadline
                if T_hat <= 1: step_success += 1
                
                # User Request: Normalize latency range 0-1 in service reward
                T_hat_clamped = min(T_hat, 1.0)
                obj = Config.training.BETA * R_total - (1 - Config.training.BETA) * T_hat_clamped
                
                grad_lambda_r = Config.training.RELIABILITY_THRESHOLD - R_total
                # Keep gradient linear for stable dual ascent, even if penalty is indicator
                grad_lambda_d = T_hat - 1.0
                
                MAX_LAMBDA = Config.training.MAX_LAMBDA
                self.lambda_r[global_idx] = np.clip(
                    self.lambda_r[global_idx] + Config.training.LR_DUAL * grad_lambda_r,
                    0.0, MAX_LAMBDA
                )
                self.lambda_d[global_idx] = np.clip(
                    self.lambda_d[global_idx] + Config.training.LR_DUAL * grad_lambda_d,
                    0.0, MAX_LAMBDA
                )
                
                # User Request: Symmetric rewards - bonus for meeting constraints, penalty for violating
                # Reliability: bonus if R > threshold, penalty if R < threshold
                reliability_margin = R_total - Config.training.RELIABILITY_THRESHOLD
                reliability_reward = self.lambda_r[global_idx] * reliability_margin
                
                # Deadline: bonus if met (T_hat <= 1), penalty if missed (T_hat > 1)
                if T_hat <= 1:
                    deadline_reward = self.lambda_d[global_idx] * (1.0 - T_hat)  # Bonus proportional to slack
                else:
                    deadline_reward = -self.lambda_d[global_idx] * 1.0  # Fixed penalty for violation
                
                agent_reward += (obj + reliability_reward + deadline_reward)
            
            rewards.append(agent_reward)

        dominant_shares = np.max(resource_loads, axis=1)
        
        sum_s = np.sum(dominant_shares)
        sum_sq_s = np.sum(dominant_shares**2)
        
        if sum_sq_s > 0:
            drf = (sum_s**2) / (self.num_servers * sum_sq_s)
        else:
            drf = 1.0
            
        grad_lambda_f = Config.training.FAIRNESS_THRESHOLD - drf
        MAX_LAMBDA_F = Config.training.MAX_LAMBDA
        self.lambda_f = np.clip(
            self.lambda_f + Config.training.LR_DUAL * grad_lambda_f,
            0.0, MAX_LAMBDA_F
        )
        
        viol_f = max(0, Config.training.FAIRNESS_THRESHOLD - drf)
        
        for i in range(self.num_servers):
            rewards[i] -= self.lambda_f * viol_f
            
        self.current_step += 1
        truncated = False
        terminated = self.current_step >= Config.system.NUM_TIME_STEPS
        
        self.last_drf = drf
        self.last_reliability = current_step_reliability
        
        self._generate_tasks()
        
        info = {
            "avg_latency": np.mean(step_latency) if step_latency else 0,
            "avg_reliability": np.mean(step_reliability) if step_reliability else 0,
            "success_rate": step_success / step_total_tasks if step_total_tasks > 0 else 0,
            "drf_fairness": drf,
            "lambda_f": self.lambda_f,
            "avg_lambda_r": np.mean(self.lambda_r),
            "avg_lambda_d": np.mean(self.lambda_d),
            "latencies": step_latency
        }
        
        return self._get_obs(), np.array(rewards), terminated, truncated, info
