import sys
import os
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecVideoRecorder

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../..')) 
sys.path.append(os.path.join(project_root, 'env'))
from env_utils import create_carracing_env 

video_folder = os.path.join(project_root, "Baselines3/SAC/video_record/video")
video_length = 1000
model_path = os.path.join(project_root, "Baselines3/SAC/model/best_model.zip")

os.makedirs(video_folder, exist_ok=True)


eval_env = create_carracing_env(render_mode="rgb_array", use_subproc=False, num_envs=1)

eval_env = VecVideoRecorder(
    eval_env,
    video_folder,
    record_video_trigger=lambda x: x == 0, 
    video_length=video_length,
    name_prefix="sac-evaluation"
)

model = SAC.load(model_path)

obs = eval_env.reset()
for _ in range(video_length): 
    action, _ = model.predict(obs, deterministic=True) 
    obs, reward, done, info = eval_env.step(action)
    if done:
        obs = eval_env.reset()

eval_env.close()

