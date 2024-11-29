import sys
import os
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback

# env path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../..'))
sys.path.append(os.path.join(project_root, 'env'))

from env_utils import create_carracing_env 

if __name__ == "__main__":
    train_env = create_carracing_env(render_mode="rgb_array", use_subproc=True, num_envs=4)

    # dir path
    model_dir = os.path.join(project_root, "Baselines3/SAC/model/")
    tensorboard_log_dir = os.path.join(project_root, "Baselines3/SAC/tensorboard/")
    eval_log_path = os.path.join(project_root, "Baselines3/SAC/eval/eval_logs/")
    best_model_dir = os.path.join(model_dir)
    last_model_dir = os.path.join(model_dir, "last_model.zip")

    # SAC 
    model = SAC(
        "CnnPolicy",
        train_env,
        verbose=1,
        buffer_size=100000,
        learning_rate=3e-4,
        batch_size=256,
        train_freq=1,
        gradient_steps=1,
        learning_starts=1000,
        gamma=0.99,
        tau=0.005,
        ent_coef="auto",
        tensorboard_log=tensorboard_log_dir  
    )

    eval_callback = EvalCallback(
        train_env,
        best_model_save_path=best_model_dir,
        log_path=eval_log_path,
        eval_freq=10000, 
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    total_timesteps = 100000
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model.save(last_model_dir)

