#!/usr/bin/env python3
"""
Calculate task arrival rate
"""

# System parameters
NUM_EDGE_SERVERS = 3
NUM_DEVICES_PER_SERVER = 5
TIME_SLOT_DURATION = 1.0  # seconds

# Calculate tasks per second
total_tasks_per_step = NUM_EDGE_SERVERS * NUM_DEVICES_PER_SERVER
tasks_per_second = total_tasks_per_step / TIME_SLOT_DURATION

print("=" * 60)
print("TASK ARRIVAL RATE ANALYSIS")
print("=" * 60)
print()
print(f"System Configuration:")
print(f"  - Number of Edge Servers: {NUM_EDGE_SERVERS}")
print(f"  - Devices per Server: {NUM_DEVICES_PER_SERVER}")
print(f"  - Time Slot Duration: {TIME_SLOT_DURATION}s")
print()
print(f"Task Generation:")
print(f"  - Tasks per time step: {total_tasks_per_step}")
print(f"  - Tasks per second: {tasks_per_second}")
print()
print(f"Load Distribution (if balanced):")
print(f"  - Tasks per server: {total_tasks_per_step / NUM_EDGE_SERVERS:.1f}")
print(f"  - Arrival rate per server: {tasks_per_second / NUM_EDGE_SERVERS:.1f} tasks/sec")
print()
print("=" * 60)
print("COMPARISON TO SERVICE RATE")
print("=" * 60)
print()

# Service rate calculation
EDGE_CPU_CAPACITY = 24e9  # 24 GHz
AVG_CPU_CYCLES = 5e9  # 5 billion cycles average
service_time = AVG_CPU_CYCLES / EDGE_CPU_CAPACITY
service_rate = 1.0 / service_time

print(f"Edge Server Capacity:")
print(f"  - CPU Capacity: {EDGE_CPU_CAPACITY/1e9:.0f} GHz")
print(f"  - Avg Task CPU: {AVG_CPU_CYCLES/1e9:.1f} billion cycles")
print(f"  - Service Time: {service_time:.3f}s per task")
print(f"  - Service Rate: {service_rate:.2f} tasks/sec")
print()

# Utilization
arrival_per_server = tasks_per_second / NUM_EDGE_SERVERS
utilization = arrival_per_server / service_rate

print(f"If tasks are evenly distributed:")
print(f"  - Arrival rate per server: {arrival_per_server:.1f} tasks/sec")
print(f"  - Utilization (ρ): {utilization:.2%}")
print(f"  Status: {'✓ Stable' if utilization < 0.9 else '⚠ High' if utilization < 1.0 else '✗ Unstable'}")
print()

# Worst case
worst_arrival = tasks_per_second  # All tasks to one server
worst_util = worst_arrival / service_rate

print(f"If all tasks go to ONE server (worst case):")
print(f"  - Arrival rate: {worst_arrival:.1f} tasks/sec")
print(f"  - Utilization (ρ): {worst_util:.2%}")
print(f"  Status: {'✓ Stable' if worst_util < 0.9 else '⚠ High' if worst_util < 1.0 else '✗ UNSTABLE - Queue will grow infinitely!'}")
print()
print("This is why load balancing with μ (forwarding) is critical!")
