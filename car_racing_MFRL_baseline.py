import os
import gymnasium as gym
from stable_baselines3 import PPO, SAC, DDPG
from stable_baselines3.common.env_util import make_vec_env


# Select the algorithm (PPO, SAC, or DDPG)
ALGORITHM = "PPO"

# Ensure the directory exists
MODEL_SAVE_DIR = "MFRL/saved_models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
LOG_DIR = f"MFRL/tensorboard_logs"
os.makedirs(LOG_DIR, exist_ok=True)


# Create the CarRacing-v2 environment
env = gym.make(
        "CarRacing-v2",
        render_mode="human",
        domain_randomize=False,
        continuous=True,
    )

# For vectorized environment
vec_env = make_vec_env("CarRacing-v2", n_envs=1)

# Define the Tensorboard log directory
log_path = os.path.join(LOG_DIR, f"{ALGORITHM}_CarRacing")

# Initialize the model
if ALGORITHM == "PPO":
    model = PPO("CnnPolicy", vec_env, verbose=1, tensorboard_log=log_path)
elif ALGORITHM == "SAC":
    model = SAC("CnnPolicy", vec_env, verbose=1, tensorboard_log=log_path)
elif ALGORITHM == "DDPG":
    model = DDPG("CnnPolicy", vec_env, verbose=1, tensorboard_log=log_path)
else:
    raise ValueError(f"Unsupported algorithm: {ALGORITHM}")

# Train the model
print("Training the model...")
model.learn(total_timesteps=100000)

# Save the trained model
save_path = os.path.join(MODEL_SAVE_DIR, f"{ALGORITHM}_CarRacing")
model.save(save_path)
print(f"Model saved to {save_path}")

# Cloase the environment
vec_env.close()

# Reload the model for evaluation (optional)
model = model.load(save_path)

# Run the trained model
obs = env.reset()[0]
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        obs = env.reset()[0]

env.close()