# 2024 RL Final Project

This repository contains implementations of various reinforcement learning algorithms.

It includes classical model-free RL algorithms **PPO**, **SAC**, and **DDPG** as baselines, 

as well as state-of-the-art model-free RL algorithms like **(TBD)** and model-based RL algorithms such as 

**Dreamer-v2**, **Dreamer-v3**, **TransDreamer**, and **TransDreamer-v2 (Ours)**.

These algorithms are applied to the `CarRacing-v2` environment from OpenAI Gymnasium's Box2D environments.

---

# Notices
**2024.12.11 (Yeonchan)**
- Dreamer-v3 configurations in configs.yaml include settings like steps, batch_size, and envs, which should be updated as needed.
- Logs and checkpoints are stored in the --logdir directory, and progress can be monitored using TensorBoard.

**2024.12.04 (Jehun)**
- Always update the **Notices** and **Updates** sections in the README when pushing any modifications
- Please add hidden files or directories to `.gitignore`.
- Do not push model files directly to the repository. Model weights will be shared separately.
- Modify code in your local branch named after yourself (e.g., `KangJehun`). Do not work directly in the main branch.
- Maintain the directory structure and adhere to the code style with sufficient comments.

# Updates
**2024.12.20 (Yeonchan)**
- Added the Dreamer test module to validate the performance of the model.
- Added a plotting utility to analyze the performance of different models. 

**2024.12.11 (Yeonchan)**
- Added Dreamer-v3 code for model-based RL experiments with CarRacing-v2.
- Updated the README to include a quick start guide for running Dreamer-v3.

**2024.12.04 (Jehun)**
- Added a wrapper to prevent the agent from leaving the track. If the agent goes too far off-track (see `env/wrapper.py`), it receives a reward penalty of -100, and the episode terminates.
- Rearranged the directory structure for easier imports in Python.
- Updated `train.py` to be compatible with three RL algorithms (SAC, PPO, DDPG). Please test the algorithm assigned to you and report any issues.
- Merged `sac_racing_play.py` and `record_sac.py` into `test.py`.

# Quick Start

### To see how the CarRacing-v2 environment works:
```bash
python3 car_racing_example.py
```

### To train with baseline models of classical RL algorithms:
```bash
cd MFRL
python3 train.py --algorithm {PPO, SAC, DDPG}
```

### To train with Dreamer-v3 model 
```bash
cd MBRL/Dreamerv3
python3 dreamer.py --configs car_racing --task car_racing_v2 --logdir ./logdir/car_racing
```

### Logs with TensorBoard

Logs are stored in: `MFRL/Baseline3/<Algorithm>/tensorborad/<Run_Name>`

ex. `MFRL/Baselines3/SAC/tensorboard/SAC_1/<log file>`

#### How to Run TensorBoard:
1) Navigate to the project directory:
   
   ```bash
   cd ~/2024_RL_Final_Project
   ```

2) Run TensorBoard for the specific folder:
   
   ```bash
   tensorboard --logdir MFRL/Baseline3/<Algorithm>/tensorborad/<Run_Name>
   ```
**Notes**: Replace 'Algorithm' and 'Run_Name' as your algorithm (SAC) and run name (ex. SAC_1) 

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
