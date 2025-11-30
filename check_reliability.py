#!/usr/bin/env python3
"""
Verify Reliability Calculation
"""
import numpy as np

# Config values
QUEUE_SENSITIVITY = 0.05
MAX_QUEUE_SIZE = 25

def calculate_reliability(queue_length, w_queue):
    exponent = -QUEUE_SENSITIVITY * queue_length * w_queue
    reliability = np.exp(exponent)
    return reliability, exponent

print("=" * 60)
print("RELIABILITY SENSITIVITY CHECK")
print("=" * 60)
print(f"QUEUE_SENSITIVITY: {QUEUE_SENSITIVITY}")
print()

scenarios = [
    (0, 0.0, "Empty Queue"),
    (1, 0.1, "Light Load"),
    (5, 0.5, "Medium Load"),
    (10, 1.0, "Heavy Load"),
    (15, 1.5, "Very Heavy Load"),
    (20, 2.0, "Congested"),
    (25, 2.5, "Full Queue")
]

print(f"{'Queue Len':<10} {'Delay (s)':<10} {'Exponent':<10} {'Reliability':<10} {'Status'}")
print("-" * 60)

for q_len, delay, status in scenarios:
    rel, exp = calculate_reliability(q_len, delay)
    print(f"{q_len:<10} {delay:<10.2f} {exp:<10.4f} {rel:<10.4f} {status}")

print()
print("=" * 60)
print("CONCLUSION")
print("=" * 60)
