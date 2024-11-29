import sys
import os
from stable_baselines3 import SAC

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../..')) 
sys.path.append(os.path.join(project_root, 'env'))

from env_utils import create_carracing_env  

model_path = os.path.join(project_root, "Baselines3/SAC/model/best_model.zip")

if __name__ == "__main__":
    eval_env = create_carracing_env(render_mode="human", use_subproc=False, num_envs=1)

    # model load
    model = SAC.load(model_path)

    obs = eval_env.reset()
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)  
        obs, reward, done, info = eval_env.step(action)
        eval_env.render()  
        if done:
            obs = eval_env.reset()  
    eval_env.close()

