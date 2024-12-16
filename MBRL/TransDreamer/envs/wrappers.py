import numpy as np
import gymnasium as gym

class OneHotAction:
    
    def __init__(self, env):
        
        self._env = env
        self.action_space = env.action_space
        
    def __getattr__(self, name):
        
        return getattr(self._env, name)
    
    def step(self, action):
        
        index = np.argmax(action).astype(int)
        reference = np.zeros_like(action)
        reference[index] = 1
        if not np.allclose(reference, action):
            raise ValueError(f"Invalid action {action}")
        return self._env.step(index)
    
    def reset(self, **kwargs):
        
        return self._env.reset(**kwargs)
    
    def sample_random_action(self):
        
        action = np.zeros(self._env.action_space.n, dtype=np.float32)
        idx = np.random.randint(0, self._env.action_space.n)
        action[idx] = 1
        return action

class Collect:

    def __init__(self, env, callbacks=None, precision=32):
        self._env = env
        self._callbacks = callbacks or []
        self._precision = precision        
        self._episode = None
    
    def __getattr__(self, name):
        return getattr(self._env, name)
    
    def step(self, action):
        
        obs, reward, terminated, truncated, info = self._env.step(action)
        
        # Process observation
        obs = {k: self._convert(v) for k, v in obs.items()}
        
        # Record transition
        transition = obs.copy()
        transition['action'] = action
        transition['reward'] = reward
        transition['discount'] = info.get('discount', np.array(1 - float(terminated or truncated)))
        transition['done'] = float(terminated or truncated) # TODO [REMINDER] will be removed
        transition['terminated'] = float(terminated)
        transition['truncated'] = float(truncated)
        self._episode.append(transition)
        
        # End of episode processing
        if terminated or truncated:
            episode = {k: [t[k] for t in self._episode] for k in self._episode[0]}
            episode = {k: self._convert(v) for k, v in episode.items()}
            info['episode'] = episode
            for callback in self._callbacks:
                callback(episode)
        obs['bev'] = obs['bev'][None,...]
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        
        obs, info = self._env.reset(**kwargs)
        obs = {k: self._convert(v) for k, v in obs.items()}
        
        # Initialize the first transition
        transition = obs.copy()
        transition['action'] = np.zeros(self._env.action_space.n, dtype=np.float32)
        transition['reward'] = 0.0
        transition['discount'] = 1.0
        transition['done'] = 0.0 # TODO [REMINDER] will be removed
        transition['terminated'] = 0.0
        transition['truncated'] = 0.0
        self._episode = [transition]
        obs['bev'] = obs['bev'][None,...]
        return obs, info
        
    def _convert(self, value):
        """ Convert value to the specified precision """
        
        value = np.array(value)
        
        if np.issubdtype(value.dtype, np.floating):
            dtype = {16: np.float16, 32: np.float32, 64: np.float64}[self._precision]
        elif np.issubdtype(value.dtype, np.signedinteger):
            dtype = {16: np.int16, 32: np.int32, 64: np.int64}[self._precision]
        elif np.issubdtype(value.dtype, np.uint8):
            dtype= np.uint8
        else:
            raise NotImplementedError(f"Unsupported dtype {value.dtype}")
        
        return value.astype(dtype)
    
class RewardObs:
    
    def __init__(self, env):
        self._env = env
    
    def __getattr__(self, name):
        return getattr(self._env, name)
    
    @property
    def observation_space(self):
        spaces = self._env.observation_space.spaces
        assert 'reward' not in spaces, "Observation space already contains 'reward'"
        spaces['reward'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(), 
                                          dtype=np.float32)
        return gym.spaces.Dict(spaces)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        obs['reward'] = np.array(reward, dtype=np.float32)
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        obs, info = self._env.reset(**kwargs)
        obs['reward'] = np.array(0.0, dtype=np.float32)
        return obs, info 
    
class TerminateOutsideTrackWrapper(gym.Wrapper):
    """
    Custom wrapper to terminate the episode if the car is too far off the track
    (i.e., the observation contains only green pixels)
    """
    
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        
        # Take a step in the environment
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Check if the car is too far away from the track
        if self._is_all_green(obs):
            terminated = True
            reward -= 100
        
        return obs, reward, terminated, truncated, info
    
    def _is_all_green(self, obs):
        """
        Check if the observation contains only green pixels
        """
        
        # Define the approximate RGB values for green
        green_rgb = np.array([100, 220, 100])
        tolerance = 10 # for detecting green pixels
        
        bev = obs['bev'][0].transpose(1, 2, 0) # (C, H, W) -> (H, W, C)        
        diff = np.abs(bev - green_rgb).mean(axis=-1)
        green_pixels = np.sum(diff < tolerance)
        
        return green_pixels > bev.shape[0] * bev.shape[1] * 0.865