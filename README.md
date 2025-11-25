# Decentralized MEC System with MAPPO and Lagrangian Constraints

This project implements a decentralized Mobile Edge Computing (MEC) system simulation where mobile devices offload tasks to edge servers. The system uses **Multi-Agent Proximal Policy Optimization (MAPPO)** to jointly optimize task offloading decisions, aiming to maximize reliability and minimize latency while satisfying fairness, reliability, and deadline constraints via **Lagrangian relaxation**.

## Features

- **MEC Environment**: A custom OpenAI Gym (Gymnasium) environment simulating:
    - Task generation with heterogeneous characteristics (size, deadline, cpu cycles).
    - M/M/1/K Queueing dynamics at Edge Servers.
    - Task partitioning (Local vs. Edge) and Forwarding (Edge vs. Neighbor).
    - Latency and Reliability modeling.
- **MAPPO Agent**:
    - Actor-Critic architecture.
    - PPO with clipping.
    - Independent learning for each Edge Server (controlling its associated devices).
- **Constrained Optimization**:
    - Lagrangian multipliers for Fairness (Jain's Index), Reliability, and Deadlines.
    - Dual ascent updates for multipliers.

## Project Structure

```
mec_system/
├── agents/
│   ├── mappo.py       # MAPPO Agent implementation
│   └── networks.py    # Actor and Critic Neural Networks
├── env/
│   ├── entities.py    # Task, MobileDevice, EdgeServer classes
│   └── mec_env.py     # Gymnasium Environment
├── config.py          # System configuration and hyperparameters
├── main.py            # Training loop
└── requirements.txt   # Dependencies
```

## Installation

1. Clone the repository (if applicable) or navigate to the project root.
2. Install dependencies:
   ```bash
   pip install -r mec_system/requirements.txt
   ```

## Usage

To run the training simulation:

```bash
# From the project root directory
python -m mec_system.main
```

This will:
1. Initialize the MEC environment with 3 Edge Servers and 5 Devices per server (default config).
2. Train the MAPPO agents for a specified number of episodes.
3. Output the training progress (Reward, Loss) to the console.
4. Save a plot of the training reward curve to `training_curve.png`.

## Configuration

You can adjust simulation parameters in `mec_system/config.py`:
- `NUM_EDGE_SERVERS`, `NUM_DEVICES_PER_SERVER`
- `NUM_TIME_STEPS` (Episode length)
- `BETA` (Weight between Reliability and Latency)
- `FAIRNESS_THRESHOLD`, `RELIABILITY_THRESHOLD`
- Learning rates and PPO hyperparameters.

## Methodology

The implementation follows a decentralized approach where each Edge Server acts as an agent making offloading decisions for its connected devices. The reward function is augmented with Lagrangian penalties to enforce:
1. **Fairness**: Ensuring resources are distributed fairly among servers (Jain's Index).
2. **Reliability**: Ensuring task success probability meets a threshold.
3. **Deadline**: Ensuring tasks complete within their maximum tolerable delay.
