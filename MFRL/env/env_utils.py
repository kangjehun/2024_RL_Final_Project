import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecTransposeImage

from .wrapper import TerminateOutsideTrackWrapper

def create_carracing_env(render_mode="rgb_array", use_subproc=False, num_envs=1):

    def make_env():
        def _init():
            env = gym.make("CarRacing-v2", render_mode=render_mode, domain_randomize=False)
            env = Monitor(env)
            env = TerminateOutsideTrackWrapper(env) # To prevent the car from driving outside the track
            return env
        return _init

    if use_subproc:
        return VecTransposeImage(SubprocVecEnv([make_env() for _ in range(num_envs)]))
    else:
        return VecTransposeImage(DummyVecEnv([make_env() for _ in range(num_envs)]))