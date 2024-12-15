import gymnasium as gym
import numpy as np

class Box2D:
    
    def __init__(self, env_name, prefix, time_limit=1000, action_repeat=4, seed=0, 
                 render_mode="rgb_array", continuous=False):
        
        if prefix == 'train':
            self._env = gym.make(env_name, render_mode=render_mode, continuous=continuous)
        else:
            self._env = gym.make(env_name, render_mode=render_mode, continuous=continuous, max_episode_steps=time_limit)
            
        self._action_repeat = action_repeat
        self._time_limit = time_limit
        self.step_count = 0
        self._env.reset(seed=seed)
        
        # Define observation space
        obs_shape = self._env.observation_space.shape # (H, W, C)
        obs_space = gym.spaces.Box(low=0, high=255, shape=(obs_shape[2], obs_shape[0], obs_shape[1]), dtype=np.uint8)
        self.observation_space = gym.spaces.Dict({'bev': obs_space})
        
        # Define action space
        self.action_space = self._env.action_space
        
    def reset(self, seed=None, options=None):
        self.step_count = 0
        obs, info = self._env.reset(seed=seed, options=options)
        obs = self._process_observation(obs)
        return {'bev': obs}, info
    
    def step(self, action):
        
        total_reward = 0
        truncated = False
        terminated = False
        info = {}
        
        for _ in range(self._action_repeat):
            
            obs, reward, terminated, truncated, info = self._env.step(action)
            total_reward += reward
            self.step_count += 1
            
            if terminated or truncated:
                break
        
        obs = self._process_observation(obs)
        return {'bev': obs}, total_reward, terminated, truncated, info
    
    def render(self, mode='rgb_array'):
        
        return self._env.render(mode=mode)
    
    def close(self):
        
        self._env.close()
    
    def _process_observation(self, obs):
        
        if len(obs.shape) == 3: 
            obs = np.transpose(obs, (2, 0, 1)) # (H, W, C) -> (C, H, W)
        else:
            raise NotImplementedError(f"Observation shape {obs.shape}")
        
        return obs
    