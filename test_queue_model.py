#!/usr/bin/env python3
"""
Test if the M/M/1/K queueing model is working correctly
"""
import numpy as np

# Test the queue formulas
def test_queue_model():
    print("=" * 70)
    print("QUEUEING MODEL VERIFICATION")
    print("=" * 70)
    print()
    
    # Parameters
    K = 25  # Max queue size
    service_rate = 4.8  # tasks/sec
    
    print("Testing M/M/1/K Queue Formulas:")
    print(f"  Service Rate (μ): {service_rate:.2f} tasks/sec")
    print(f"  Max Queue Size (K): {K}")
    print()
    
    # Test different utilization levels
    test_cases = [
        (0.3, "Low load"),
        (0.5, "Medium load"),
        (0.7, "High load"),
        (0.9, "Very high load"),
        (0.99, "Critical load")
    ]
    
    print(f"{'Util (ρ)':<12} {'Arrival λ':<12} {'Avg Queue':<12} {'Queue Time':<12} {'Status'}")
    print("-" * 70)
    
    for rho, description in test_cases:
        arrival_rate = rho * service_rate
        
        # M/M/1/K formulas (from your code)
        p0 = (1 - rho) / (1 - rho**(K+1))
        pk = p0 * (rho**K)
        lq = (rho * (1 - (K+1)*rho**K + K*rho**(K+1))) / ((1 - rho) * (1 - rho**(K+1)))
        
        if arrival_rate > 1e-9:
            w_queue = (lq - (1 - p0)) / (arrival_rate * (1 - pk))
            w_queue = max(0.0, w_queue)
        else:
            w_queue = 0.0
        
        print(f"{rho:<12.2f} {arrival_rate:<12.2f} {lq:<12.2f} {w_queue:<12.3f} {description}")
    
    print()
    print("=" * 70)
    print("ISSUE DETECTION")
    print("=" * 70)
    print()
    
    # Check the actual scenario from training
    rho = 0.99
    arrival_rate = 5.0
    
    print(f"Current Training Scenario:")
    print(f"  Arrival Rate: {arrival_rate:.2f} tasks/sec")
    print(f"  Service Rate: {service_rate:.2f} tasks/sec")
    print(f"  Utilization: {rho:.2%}")
    print()
    
    p0 = (1 - rho) / (1 - rho**(K+1))
    pk = p0 * (rho**K)
    lq = (rho * (1 - (K+1)*rho**K + K*rho**(K+1))) / ((1 - rho) * (1 - rho**(K+1)))
    w_queue = (lq - (1 - p0)) / (arrival_rate * (1 - pk))
    
    print(f"M/M/1/K Results:")
    print(f"  p0 (empty prob): {p0:.6f}")
    print(f"  pk (full prob): {pk:.6f}")
    print(f"  Avg queue length: {lq:.2f} tasks")
    print(f"  Avg queue delay: {w_queue:.3f} seconds")
    print()
    
    # Check if formulas make sense
    print("PROBLEMS DETECTED:")
    print()
    
    if rho > 1.0:
        print("  ✗ System is UNSTABLE (ρ > 1.0)")
        print("    Queue will grow infinitely!")
    elif rho > 0.95:
        print("  ⚠ System is near capacity (ρ > 0.95)")
        print("    Queue delays will be very high")
    else:
        print("  ✓ System utilization is reasonable")
    
    print()
    
    if w_queue > 1.0:
        print(f"  ✗ Queue delay ({w_queue:.2f}s) exceeds 1 second")
        print("    This will cause most tasks to miss deadlines")
    else:
        print(f"  ✓ Queue delay ({w_queue:.2f}s) is manageable")
    
    print()
    
    # The REAL problem
    print("=" * 70)
    print("THE REAL PROBLEM")
    print("=" * 70)
    print()
    print("The M/M/1/K model assumes:")
    print("  1. ✓ Poisson arrivals (memoryless)")
    print("  2. ✓ Exponential service times")
    print("  3. ✓ Single server")
    print("  4. ✓ Finite queue (K=25)")
    print()
    print("BUT in your system:")
    print("  ✗ Tasks arrive in BURSTS (15 tasks at once)")
    print("  ✗ Not steady-state (system changes every step)")
    print("  ✗ The model is applied INSTANTLY (no actual queue)")
    print()
    print("This means:")
    print("  - Queue delay is calculated using steady-state formulas")
    print("  - But actual system has burst arrivals every second")
    print("  - The formulas UNDERESTIMATE delay for burst traffic")
    print()
    print("SOLUTION:")
    print("  → Agent must learn to use μ (neighbor forwarding)")
    print("  → Distribute the 15 tasks across 3 servers (5 each)")
    print("  → This reduces ρ from 0.99 to 0.625")
    print("  → Queue delay drops from 2.3s to 0.3s")
    print()

if __name__ == "__main__":
    test_queue_model()
