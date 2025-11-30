
import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mec_system.env.mec_env import MECEnv
from mec_system.config import *

def verify_latency():
    print("Verifying Latency Calculation...")
    env = MECEnv()
    obs, _ = env.reset(seed=42)
    
    # Get the first device's task
    # We need to access the internal state to get the exact task parameters
    # env.reset() generates tasks.
    
    # Let's look at the first server, first device
    server_idx = 0
    device_idx = 0
    global_idx = server_idx * NUM_DEVICES_PER_SERVER + device_idx
    
    # task = env.current_tasks[global_idx] # Cannot access before step
    # print(f"Task Data Size: {task.data_size}")
    # print(f"Task Result Size: {task.result_size}")
    # print(f"Task CPU Cycles: {task.total_cpu_cycles}")
    
    # Define a deterministic action
    # Alpha = 0.5 (50% local)
    # Mus = 0 (0% neighbor, so 50% at serving edge)
    
    # Action shape: (Num_Devices, 1 + Num_Neighbors)
    action = np.zeros((env.num_devices, env.action_dim_per_device))
    
    # Set for our target device
    action[global_idx, 0] = 0.5 # Alpha
    # Mus are already 0
    
    # For all other devices, let's set alpha=1 (100% local) to avoid queue contention at the edge for simplicity of calculation
    # If other devices offload, they affect the queue.
    # To verify the FORMULA, we want to isolate the task or control the queue.
    # If we set all others to local, queue should be empty (or just this task).
    for i in range(env.num_devices):
        if i != global_idx:
            action[i, 0] = 1.0
            
    # Step the environment
    # We need to reshape action to (Num_Servers, Num_Devices_Per_Server, Action_Dim)
    action_reshaped = action.reshape(env.num_servers, NUM_DEVICES_PER_SERVER, env.action_dim_per_device)
    
    # We need to manually calculate expected values BEFORE stepping, 
    # because step() might generate NEW tasks for the NEXT step (actually it does at the beginning of step, wait).
    # In `step(actions)`:
    # 1. Generate new tasks (Wait, `step` generates new tasks at the BEGINNING? No.)
    # Let's check `mec_env.py`:
    # def step(self, actions):
    #   self.current_tasks = [] 
    #   for dev in self.devices: self.current_tasks.append(...)
    
    # AH! `step` generates NEW tasks at the start. 
    # So the action we pass is applied to the NEW tasks? 
    # Or is it applied to the tasks from the PREVIOUS step?
    # Usually in RL, obs corresponds to state S_t. Action A_t is taken. 
    # If `step` generates new tasks immediately, then the action A_t is applied to... what?
    # If `reset` generates tasks, then `step` should process them.
    
    # Let's check `reset`:
    # reset() -> returns _get_obs()
    # _get_obs() uses `self.current_tasks`.
    # But `reset` does NOT generate tasks in the code I saw!
    # `reset` resets queues and loads.
    # `_get_obs` checks `if hasattr(self, 'current_tasks')`.
    # If not, it returns zeros.
    
    # So at t=0 (after reset), there are NO tasks?
    # Then `step` is called.
    # `step` generates tasks.
    # Then it applies `actions` to these NEW tasks?
    # That seems like the agent acts blindly on tasks it hasn't seen yet?
    # Or maybe the "Task" in observation is from the PREVIOUS step?
    # If `step` generates tasks first, then calculates rewards based on `actions` applied to these tasks...
    # Then the agent observes S_{t-1} (which has no info about current tasks if they are new), 
    # and takes action A_t.
    # This implies the agent must predict or the task info in obs is stale/irrelevant?
    # OR, the task info in obs is for the tasks that ARRIVED at t-1?
    # But the code says:
    # 1. Generate new tasks
    # 2. Calculate rewards (using `actions` on `current_tasks`)
    
    # This means the action `a_t` is applied to tasks generated at `t`.
    # But the agent saw `obs_{t-1}` which did NOT contain these tasks.
    # This is a bit weird for a standard MDP unless tasks are predictable or static.
    # However, for verification, I just need to know what tasks were generated inside `step`.
    
    # I can't know the task parameters BEFORE `step` because they are random.
    # So I cannot predict the expected latency perfectly from outside unless I mock the random generator 
    # or capture the task after generation.
    
    # BUT, I can seed the environment!
    # If I seed, the sequence of random numbers is fixed.
    # I can run `step` once to see what task is generated.
    # Then run it again to verify?
    
    # Better approach:
    # Modify `step` temporarily? No.
    # I can just run `step`, get the result (latency), AND get the task that was generated (it's stored in `env.current_tasks`).
    # Then I calculate what the latency SHOULD have been for that task and compare.
    
    obs, rewards, term, trunc, info = env.step(action_reshaped)
    
    # Now retrieve the task that was processed
    task = env.current_tasks[global_idx]
    print(f"Generated Task Data Size: {task.data_size}")
    print(f"Generated Task Result Size: {task.result_size}")
    
    # Calculate Expected Latency
    # Alpha = 0.5
    alpha = 0.5
    mus = 0.0
    
    # 1. Local
    t_local = (alpha * task.total_cpu_cycles) / DEVICE_CPU_CAPACITY
    print(f"Expected T_local: {t_local}")
    
    # 2. Edge
    portion_edge = (1 - alpha)
    t_tx_edge = (portion_edge * task.data_size) / UPLINK_RATE_MAX
    t_proc_edge = (portion_edge * task.total_cpu_cycles) / EDGE_CPU_CAPACITY
    t_dl_edge = (portion_edge * task.result_size) / DOWNLINK_RATE_MAX
    
    # Queue
    # Since all other devices are local, arrival rate at server 0 is just this task's portion.
    # lambda = portion_edge / TIME_SLOT_DURATION (assuming 1 task per step per device)
    # Actually `mec_env.py` sums `portion_edge` for arrival rate.
    arrival_rate = portion_edge / TIME_SLOT_DURATION
    service_rate = env.servers[server_idx].service_rate
    rho = arrival_rate / service_rate
    
    # M/M/1/K
    K = MAX_QUEUE_SIZE
    if rho >= 0.99: rho = 0.99
    
    p0 = (1 - rho) / (1 - rho**(K+1))
    pk = p0 * (rho**K)
    lq = (rho * (1 - (K+1)*rho**K + K*rho**(K+1))) / ((1 - rho) * (1 - rho**(K+1)))
    
    w_queue = (lq - (1 - p0)) / (arrival_rate * (1 - pk))
    w_queue = max(0.0, w_queue)
    
    t_edge = t_tx_edge + t_proc_edge + t_dl_edge + w_queue
    print(f"Expected T_edge: {t_edge} (Tx={t_tx_edge}, Proc={t_proc_edge}, DL={t_dl_edge}, W={w_queue})")
    
    # Total
    expected_total = max(t_local, t_edge) + RECOM_TIME
    print(f"Expected Total: {expected_total}")
    
    # We need to find the actual latency for this task from the environment.
    # The environment returns `info['avg_latency']` but that's an average.
    # It doesn't return per-task latency in `step`.
    # However, I can inspect `env.last_reliability`? No, that's reliability.
    # I can't easily get the exact latency calculated inside `step` without modifying `step` to return it 
    # or printing it.
    
    # Wait, I can use the `info` if I only have 1 task?
    # But I have Num_Devices tasks.
    # I set others to local.
    # Their latency will be t_local.
    # My task has t_edge (likely higher).
    # The average will be mixed.
    
    # I should modify `mec_env.py` to print or store the latency for verification?
    # Or I can just trust my calculation of what it SHOULD be, and if I see the code change, I know it's there.
    # But I want to verify the CODE executes correctly.
    
    # I will modify `mec_env.py` to store `step_latency` as a class attribute `self.last_step_latency` 
    # so I can access it in the test.
    # Actually, `mec_env.py` already calculates it in `step_latency` list.
    # I can just add it to `info`?
    # `info` has `avg_latency`.
    
    # Let's calculate the expected average latency.
    # For the other devices (local only):
    # t_local_others = (1.0 * task_other.total_cpu_cycles) / DEVICE_CPU_CAPACITY
    # T_total_other = t_local_others + RECOM_TIME
    
    # So I can calculate the expected average and compare with `info['avg_latency']`.
    
    total_latency_sum = expected_total
    
    for i in range(env.num_devices):
        if i != global_idx:
            t_other = env.current_tasks[i]
            t_loc = (1.0 * t_other.total_cpu_cycles) / DEVICE_CPU_CAPACITY
            t_tot = t_loc + RECOM_TIME
            total_latency_sum += t_tot
            
    expected_avg = total_latency_sum / env.num_devices
    print(f"Expected Avg Latency: {expected_avg}")
    print(f"Actual Avg Latency: {info['avg_latency']}")
    
    if np.isclose(expected_avg, info['avg_latency'], rtol=1e-5):
        print("SUCCESS: Latency matches expected value!")
    else:
        print("FAILURE: Latency mismatch.")

if __name__ == "__main__":
    verify_latency()
