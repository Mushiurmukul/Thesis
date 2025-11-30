#!/usr/bin/env python3
"""
Analyze queueing delay in isolation from other latency components
"""
import numpy as np

# System parameters
NUM_SERVERS = 3
TIME_SLOT_DURATION = 1.0  # seconds
MAX_QUEUE_SIZE = 25
EDGE_CPU_CAPACITY = 24e9  # 24 GHz
NUM_TASKS_PER_STEP = 15  # 3 servers × 5 devices

# Typical task parameters (from your data)
AVG_CPU_CYCLES = 5e9  # 5 billion cycles (average)
SERVICE_TIME = AVG_CPU_CYCLES / EDGE_CPU_CAPACITY  # ~0.21 seconds
SERVICE_RATE = 1.0 / SERVICE_TIME  # ~4.76 tasks/second

print("=" * 70)
print("QUEUEING DELAY ANALYSIS")
print("=" * 70)
print()
print("System Parameters:")
print(f"  - Edge CPU Capacity: {EDGE_CPU_CAPACITY/1e9:.1f} GHz")
print(f"  - Max Queue Size (K): {MAX_QUEUE_SIZE}")
print(f"  - Time Slot Duration: {TIME_SLOT_DURATION}s")
print(f"  - Tasks per time step: {NUM_TASKS_PER_STEP}")
print(f"  - Service Rate (μ): {SERVICE_RATE:.2f} tasks/sec")
print(f"  - Service Time: {SERVICE_TIME:.3f} seconds")
print()

def calculate_queue_delay(arrival_rate, service_rate, K):
    """Calculate M/M/1/K queue delay"""
    rho = arrival_rate / service_rate
    if rho >= 0.99:
        rho = 0.99
    
    if rho >= 1.0:
        return float('inf')  # Unstable system
    
    # M/M/1/K formulas
    p0 = (1 - rho) / (1 - rho**(K+1))
    pk = p0 * (rho**K)
    lq = (rho * (1 - (K+1)*rho**K + K*rho**(K+1))) / ((1 - rho) * (1 - rho**(K+1)))
    
    if arrival_rate > 1e-9:
        w_queue = (lq - (1 - p0)) / (arrival_rate * (1 - pk))
        w_queue = max(0.0, w_queue)
    else:
        w_queue = 0.0
    
    return w_queue, lq, rho

print("SCENARIO ANALYSIS:")
print("-" * 70)
print()

# Scenario 1: Balanced load (5 tasks per server)
print("Scenario 1: BALANCED LOAD (5 tasks/server)")
arr_rate_balanced = (NUM_TASKS_PER_STEP / NUM_SERVERS) / TIME_SLOT_DURATION  # 5 tasks/sec
w_queue, lq, rho = calculate_queue_delay(arr_rate_balanced, SERVICE_RATE, MAX_QUEUE_SIZE)
print(f"  Arrival Rate: {arr_rate_balanced:.2f} tasks/sec")
print(f"  Utilization (ρ): {rho:.3f} ({rho*100:.1f}%)")
print(f"  Avg Queue Length: {lq:.2f} tasks")
print(f"  Queue Delay: {w_queue:.3f} seconds")
print(f"  → This means ~{w_queue*1000:.0f}ms delay just from waiting in queue!")
print()

# Scenario 2: Imbalanced load (agent sends 10 tasks to one server)
print("Scenario 2: IMBALANCED LOAD (10 tasks to one server)")
arr_rate_heavy = 10 / TIME_SLOT_DURATION  # 10 tasks/sec
w_queue, lq, rho = calculate_queue_delay(arr_rate_heavy, SERVICE_RATE, MAX_QUEUE_SIZE)
print(f"  Arrival Rate: {arr_rate_heavy:.2f} tasks/sec")
print(f"  Utilization (ρ): {rho:.3f} ({rho*100:.1f}%)")
print(f"  Avg Queue Length: {lq:.2f} tasks")
print(f"  Queue Delay: {w_queue:.3f} seconds")
print(f"  → This means ~{w_queue*1000:.0f}ms delay just from waiting in queue!")
print()

# Scenario 3: Light load (agent forwards 3 tasks to neighbor)
print("Scenario 3: LIGHT LOAD (3 tasks to one server)")
arr_rate_light = 3 / TIME_SLOT_DURATION  # 3 tasks/sec
w_queue, lq, rho = calculate_queue_delay(arr_rate_light, SERVICE_RATE, MAX_QUEUE_SIZE)
print(f"  Arrival Rate: {arr_rate_light:.2f} tasks/sec")
print(f"  Utilization (ρ): {rho:.3f} ({rho*100:.1f}%)")
print(f"  Avg Queue Length: {lq:.2f} tasks")
print(f"  Queue Delay: {w_queue:.3f} seconds")
print(f"  → This means ~{w_queue*1000:.0f}ms delay just from waiting in queue!")
print()

print("=" * 70)
print("TOTAL LATENCY BREAKDOWN (Example Task)")
print("=" * 70)
print()
print("For a typical task with balanced load:")
print(f"  1. Queue Delay:       {w_queue:.3f}s  ({w_queue/(w_queue+SERVICE_TIME+0.15)*100:.1f}% of total)")
print(f"  2. Processing Time:   {SERVICE_TIME:.3f}s  ({SERVICE_TIME/(w_queue+SERVICE_TIME+0.15)*100:.1f}% of total)")
print(f"  3. Transmission Time: 0.150s  (15.0% of total)")
print(f"  ---")
print(f"  TOTAL LATENCY:        {w_queue+SERVICE_TIME+0.15:.3f}s")
print()

# Calculate what happens with current arrival rate
print("=" * 70)
print("ACTUAL SYSTEM BEHAVIOR")
print("=" * 70)
print()
arr_rate_balanced = 5.0  # Based on observed data
w_queue, lq, rho = calculate_queue_delay(arr_rate_balanced, SERVICE_RATE, MAX_QUEUE_SIZE)
total_latency = w_queue + SERVICE_TIME + 0.15

print(f"Current Arrival Rate: {arr_rate_balanced:.2f} tasks/sec")
print(f"Queue Delay: {w_queue:.3f}s")
print(f"Processing + Transmission: {SERVICE_TIME + 0.15:.3f}s")
print(f"TOTAL: {total_latency:.3f}s")
print()
print(f"Deadline: 2.2s (avg)")
print(f"Success: {'✓ MEETS' if total_latency < 2.2 else '✗ MISSES'} deadline")
print()
if total_latency > 2.2:
    print(f"Overage: {total_latency - 2.2:.3f}s ({(total_latency - 2.2)/2.2*100:.1f}% over deadline)")
    print()
    print("ROOT CAUSE: Queue delay is the dominant factor!")
    print(f"  - Queue contributes: {w_queue/total_latency*100:.1f}% of total latency")
    print(f"  - To meet deadline, need queue delay < {2.2 - SERVICE_TIME - 0.15:.3f}s")
    required_rho = 0.5  # Target utilization
    required_arrival = required_rho * SERVICE_RATE
    print(f"  - This requires reducing arrival rate to ~{required_arrival:.1f} tasks/sec")
    print(f"  - Or using neighbor forwarding (μ) to distribute load")
