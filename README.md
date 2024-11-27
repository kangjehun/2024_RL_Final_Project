# 2024 RL Final Project

This repository contains implementations of various reinforcement learning algorithms.

It includes classical model-free RL algorithms such as **PPO**, **SAC**, and **DDPG** as baselines, 

as well as state-of-the-art model-free RL algorithms like **(TBD)** and model-based RL algorithms such as 

**Dreamer-v2**, **Dreamer-v3**, **TransDreamer**, and **TransDreamer-v2 (Ours)**.

These algorithms are applied to the `CarRacing-v2` environment from OpenAI Gymnasium's Box2D environments.

---

# Quick Start

### To see how the CarRacing-v2 environment works:
```bash
python3 car_racing_example.py
```

### To train with baseline models of classical RL algorithms:
```bash
python3 car_racing_MFRL_baseline.py
```

### Logs with TensorBoard

Logs are stored in: `MFRL/tensorboard_logs/<Algorithm>/<Run_Name>`

ex. `MFRL/tensorboard_logs/PPO_CarRacing/PPO_1`

#### How to Run TensorBoard:
1) Navigate to the project directory:
   
   ```bash
   cd ~/2024_RL_Final_Project
   ```

2) Run TensorBoard for the specific folder:
   
   ```bash
   tensorboard --logdir MFRL/tensorboard_logs/PPO_CarRacing/PPO_1
   ```
**Notes**: Replace `PPO_1` with the folder name of the run you want to view.

3) Open the browser and go to the address shown (e.g., `http://localhost:6006/`).

---

### commands (temp)

Check installed GPU Devices

```bash
lspci | grep -i nvidia
```

Check GPU status during training

```bash
watch -n 1 nvidia-smi
```

