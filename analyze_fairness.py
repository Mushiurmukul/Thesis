#!/usr/bin/env python3
"""
Analyze if edge servers are treated fairly
"""
import numpy as np

# Simulate DRF fairness calculation
def calculate_drf(resource_loads):
    """
    resource_loads: shape (num_servers, num_resources)
    Each row is a server, each column is a resource type (CPU, BW, Storage)
    """
    # Dominant share = max resource utilization per server
    dominant_shares = np.max(resource_loads, axis=1)
    
    sum_s = np.sum(dominant_shares)
    sum_sq_s = np.sum(dominant_shares**2)
    
    if sum_sq_s > 0:
        drf = (sum_s**2) / (len(resource_loads) * sum_sq_s)
    else:
        drf = 1.0
    
    return drf, dominant_shares

print("=" * 70)
print("EDGE SERVER FAIRNESS ANALYSIS")
print("=" * 70)
print()

# Test different load distributions
print("Scenario 1: PERFECTLY BALANCED (Equal load on all servers)")
print("-" * 70)
balanced_loads = np.array([
    [0.33, 0.30, 0.25],  # Server 0: CPU=33%, BW=30%, Storage=25%
    [0.33, 0.30, 0.25],  # Server 1: CPU=33%, BW=30%, Storage=25%
    [0.33, 0.30, 0.25]   # Server 2: CPU=33%, BW=30%, Storage=25%
])
drf, shares = calculate_drf(balanced_loads)
print(f"Resource Loads:")
for i, load in enumerate(balanced_loads):
    print(f"  Server {i}: CPU={load[0]:.2f}, BW={load[1]:.2f}, Storage={load[2]:.2f} → Dominant={shares[i]:.2f}")
print(f"\nDRF Fairness: {drf:.4f}")
print(f"Interpretation: {drf:.4f} = {'PERFECT' if drf > 0.99 else 'GOOD' if drf > 0.9 else 'FAIR' if drf > 0.8 else 'POOR'}")
print()

print("Scenario 2: IMBALANCED (One server overloaded)")
print("-" * 70)
imbalanced_loads = np.array([
    [0.80, 0.70, 0.60],  # Server 0: HEAVY load
    [0.20, 0.15, 0.10],  # Server 1: Light load
    [0.20, 0.15, 0.10]   # Server 2: Light load
])
drf, shares = calculate_drf(imbalanced_loads)
print(f"Resource Loads:")
for i, load in enumerate(imbalanced_loads):
    print(f"  Server {i}: CPU={load[0]:.2f}, BW={load[1]:.2f}, Storage={load[2]:.2f} → Dominant={shares[i]:.2f}")
print(f"\nDRF Fairness: {drf:.4f}")
print(f"Interpretation: {drf:.4f} = {'PERFECT' if drf > 0.99 else 'GOOD' if drf > 0.9 else 'FAIR' if drf > 0.8 else 'POOR'}")
print()

print("Scenario 3: MODERATELY IMBALANCED")
print("-" * 70)
moderate_loads = np.array([
    [0.60, 0.50, 0.40],  # Server 0: High load
    [0.40, 0.30, 0.25],  # Server 1: Medium load
    [0.30, 0.20, 0.15]   # Server 2: Low load
])
drf, shares = calculate_drf(moderate_loads)
print(f"Resource Loads:")
for i, load in enumerate(moderate_loads):
    print(f"  Server {i}: CPU={load[0]:.2f}, BW={load[1]:.2f}, Storage={load[2]:.2f} → Dominant={shares[i]:.2f}")
print(f"\nDRF Fairness: {drf:.4f}")
print(f"Interpretation: {drf:.4f} = {'PERFECT' if drf > 0.99 else 'GOOD' if drf > 0.9 else 'FAIR' if drf > 0.8 else 'POOR'}")
print()

print("=" * 70)
print("HOW DRF WORKS")
print("=" * 70)
print()
print("DRF (Dominant Resource Fairness) measures how evenly resources")
print("are distributed across servers:")
print()
print("  1. Find dominant share for each server (max resource usage)")
print("  2. Calculate Jain's Fairness Index:")
print("     DRF = (sum of shares)² / (num_servers × sum of shares²)")
print()
print("DRF Range:")
print("  1.0 = Perfect fairness (all servers equally loaded)")
print("  0.33 = One server does all work, others idle")
print()
print("YOUR SYSTEM:")
print(f"  Fairness Threshold: 0.80")
print(f"  Current Training: ~0.78 (below threshold)")
print()
print("This means:")
print("  ✗ Servers are NOT equally loaded")
print("  ✗ Some servers are overloaded while others are idle")
print("  ✗ Agent needs to learn better load distribution")
print()
print("Goal: Agent should use μ (forwarding) to balance load")
print("      and achieve DRF > 0.80")
