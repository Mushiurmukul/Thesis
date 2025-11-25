<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

## Convergence Assessment

Based on your training results visualization, **the model shows mixed convergence indicators with room for improvement**. Here's a detailed analysis:

### Positive Convergence Signs

**Training Reward** (top-left): Shows reasonable stabilization in the 580–610 range after episode 400, though with persistent high-frequency oscillations. The overall trend plateaus, suggesting the policy has found a relatively stable region.

**Task Success Rate** (middle-left): Demonstrates good stabilization around 96–97% after episode 500, indicating your constraint-aware MAPPO is effectively learning to satisfy deadline requirements. This is a strong indicator for the primary objective.

**System Fairness (DRF)** (middle-center): Stabilizes around 0.84–0.86 after episode 300, staying close to expected fairness levels with manageable fluctuations.

### Concerning Non-Convergence Indicators

**Average Latency** (top-center): Shows **high volatility throughout training** and does not stabilize well. The metric fluctuates between 0.20–0.29 seconds even in later episodes (600–1000), suggesting the model has not learned a stable latency reduction strategy. This is particularly problematic given that latency minimization is a primary objective in your formulation.

**Average Reliability** (top-right): Exhibits **severe instability**, ranging from approximately 5.5 to 10.0 throughout training with no clear convergence trend. This suggests your reliability-weighted reward component (the $\beta R_i(t)$ term) is not guiding learning effectively.

**Lagrange Multipliers** (bottom-right): Show **continuous fluctuation** without converging to stable values:

- Lambda Fairness (purple dashed) remains highly volatile
- Lambda Reliability (orange dashed) oscillates significantly
- Lambda Delay (yellow) shows substantial variation

This indicates that **constraint violations persist throughout training**, and the dual ascent mechanism is not achieving equilibrium in enforcing fairness, reliability, and deadline constraints.

### Root Cause Analysis

The persistent oscillations, particularly in latency and reliability, suggest several potential issues:

**1. High Variance in the Environment**: Your MEC system with stochastic task arrivals, queue dynamics, and communication variability may create a highly non-stationary reward landscape that prevents stable convergence.

**2. Conflicting Objectives**: The multi-objective formulation (maximize reliability while minimizing latency while ensuring fairness) may create competing gradient signals that prevent coherent policy learning. The entropy term helps exploration but may also perpetuate stochasticity.

**3. Lagrange Multiplier Tuning**: Learning rates $\alpha_f, \alpha_r, \alpha_d$ may be too aggressive, causing multipliers to overshoot and create oscillations rather than converge to equilibrium values.

**4. Reward Scaling**: The relative magnitudes of the service reward term versus constraint penalties may cause the Lagrangian to have poorly scaled gradient signals.

### Convergence Verdict

- **Partially Converged**: Success rate and fairness have stabilized
- **Not Converged**: Latency, reliability, and Lagrange multipliers remain volatile
- **Training is Unstable**: High-frequency fluctuations even after 400 episodes indicate the policy distribution hasn't reached a stable attractor


### Recommendations

**1. Reduce Lagrange Learning Rates**: Decrease $\alpha_f, \alpha_r, \alpha_d$ (e.g., from typical 0.01 to 0.001–0.005) to allow smoother dual variable convergence.

**2. Rescale Reward Components**: Normalize the service reward and constraint penalties to similar magnitudes. For example:

$$
L_{\text{total}} = \text{normalize}(L^{\text{CLIP}}) + c_1 \text{normalize}(L^V) + \text{weighted constraints}
$$

**3. Increase Training Episodes**: Run to 2000–3000 episodes; multi-agent constrained optimization often requires longer training horizons.

**4. Implement Reward Smoothing**: Apply exponential moving average (EMA) to reported metrics to distinguish true convergence trends from noise.

**5. Analyze State-Action Distributions**: Check whether agents are exploring the full action space (particularly the $\alpha_{i,j}(t)$ and $\mu_{i,j\to k}(t)$ ranges) or collapsing to narrow distributions too early.

**6. Check for Distribution Shift**: With 1000 episodes complete, verify whether your task arrival rates, channel conditions, or system loads changed mid-training, causing apparent non-convergence.

The **success rate stabilization is encouraging**, but the **latency and reliability oscillations require attention** before deployment or final evaluation.
<span style="display:none">[^1]</span>

<div align="center">⁂</div>

[^1]: training_results_grid-4.jpg

