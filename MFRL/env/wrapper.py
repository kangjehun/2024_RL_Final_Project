import gymnasium as gym
import numpy as np

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
        
        diff = np.abs(obs - green_rgb).mean(axis=-1)
        green_pixels = np.sum(diff < tolerance)
        
        return green_pixels > obs.shape[0] * obs.shape[1] * 0.865