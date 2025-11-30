<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# \section{Dataset Parameters}

- Input data size (bits)
- CPU cycles required per bit
- Task deadline (seconds)
- CPU capabilities of device, local MEC server, and neighboring servers (cycles/second)
- Transmission rates (bits/second)
- Result size to be sent back (bits)
- Channel error rate parameter
- : Task vulnerability factor
- Fault rates of processors at different CPU frequencies
- Reliability threshold
- : Minimum fairness threshold
- : Number of time slots in the simulation

Based on recent 2024-2025 research papers (specifically *Liu et al. 2025*, *Jin et al. 2024*, and *Iqbal et al. 2025*), here are realistic simulation parameters for your dataset.

### **Core System Parameters (2025 Standards)**

| Parameter | Value Range / Typical Value | Source / Context |
| :-- | :-- | :-- |
| **Input Data Size** | **0.5 MB – 5 MB** (4 - 40 Mbits) | Typical for video frames, sensing data, or AR packets [^1][^2]. |
| **CPU Cycles per Bit** | **500 – 2,500 cycles/bit** | Depends on complexity (e.g., 500 for simple data processing, 2500 for DNN inference) [^1]. |
| **Task Deadline** | **0.1 s – 1.0 s** (100ms - 1000ms) | Strict URLLC apps use 10-50ms; general IoV/MEC apps use 100ms-1s [^3]. |
| **Result Size** | **10% – 20% of input size** | Typically 0.05 MB – 0.5 MB (Control signals or inference labels are small) [^4]. |
| **Transmission Rate** | **10 Mbps – 100 Mbps** | V2I / Device-to-Edge 5G uplink rates [^2]. |

### **Computational Capabilities**

| Device Type | CPU Frequency (Cycles/sec) | Notes |
| :-- | :-- | :-- |
| **Local Device (IoT/Phone)** | **0.5 GHz – 1.0 GHz** | Limited power, intended for light pre-processing [^1]. |
| **Local MEC Server** | **3.0 GHz – 5.0 GHz** (per core) | Often modeled as total capacity: **10 - 20 G cycles/s** [^5]. |
| **Neighboring Server** | **3.0 GHz – 5.0 GHz** | Same class as local MEC, connected via high-speed fiber backhaul [^4]. |

### **Reliability \& Fault Parameters**

| Parameter | Value | Context |
| :-- | :-- | :-- |
| **Channel Error Rate** | **$10^{-5}$ – $10^{-3}$** | Packet error rate (PER) in wireless 5G/6G channels [^3]. |
| **Task Vulnerability Factor** | **0.001 – 0.01** | Probability that a task fails due to transient hardware faults during execution. |
| **Processor Fault Rate** | $\lambda_{fault} \approx 10^{-4} \text{ to } 10^{-3} \text{ / sec}$ | Faults scale with DVFS frequency ($f$): often modeled as $\lambda_0 e^{c f}$ [^3]. |
| **Reliability Threshold** | **95% – 99.9% (0.95 - 0.999)** | URLLC demands 99.999%, but general MEC optimization targets ~99% [^3]. |

### **Other Simulation Settings**

| Parameter | Value |
| :-- | :-- |
| **Fairness Threshold ($\epsilon$)** | **0.7 – 0.8 (Jain’s Index)** |
| **Time Slots ($T$)** | **500 – 2,000 slots** |
| **Bandwidth** | **10 MHz – 20 MHz** |

### **Recommendation for Your Paper**

"We adopt simulation parameters aligned with recent MEC studies. Tasks are generated with input sizes uniformly distributed between **0.5–5 Mbits**, requiring **1000 cycles/bit** on average. The local edge server operates at **10 GHz** (total capacity), while user devices have **1 GHz** capacity. We set the task deadline to **500 ms**, with a target reliability threshold of **99%** and a fairness threshold (Jain's Index) of **0.8**."[^6][^7]

* Y. Liu et al., "Task Offloading... for MEC-assisted IoV," 2025.*[^6]
* A. Iqbal et al., "Energy-Aware Ultra-Reliable Low-Latency...," 2025.*[^7]
<span style="display:none">[^10][^11][^12][^8][^9]</span>

<div align="center">⁂</div>

[^1]: https://www.nature.com/articles/s41598-024-79464-2

[^2]: https://www.scitepress.org/Papers/2025/136482/136482.pdf

[^3]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12158281/

[^4]: https://www.arxiv.org/pdf/2508.15795.pdf

[^5]: http://www.csroc.org.tw/journal/JOC35-6/JOC3506-05.pdf

[^6]: training_results_grid-6.jpg

[^7]: https://docs.pytorch.org/rl/0.4/tutorials/multiagent_ppo.html

[^8]: https://clouds.cis.unimelb.edu.au/papers/TaskOffloading-CC.pdf

[^9]: https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2025.1538277/full

[^10]: https://www.thinkmind.org/articles/cloud_computing_2025_2_50_20029.pdf

[^11]: https://www.sciencedirect.com/science/article/pii/S2542660524003263

[^12]: https://www.sciencedirect.com/org/science/article/pii/S1548771725000193

