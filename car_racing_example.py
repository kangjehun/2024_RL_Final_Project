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
        print(f"Green pixels: {green_pixels}")
        print(f"Obs shape: {obs.shape}")
        
        return green_pixels > obs.shape[0] * obs.shape[1] * 0.865
        

def forward_acceleration_action():
    """
    Generate an action that applies only forward acceleration
    (no steering, half throttle, no brake)
    """
    return np.array([0.0, 0.5, 0.0])

def main():
    
    # Create the CarRacing-v2 environment
    env = gym.make(
        "CarRacing-v2",
        render_mode="human",
        domain_randomize=False,
        continuous=True,
    )
    
    # Wrap the environment with the custom wrapper
    env = TerminateOutsideTrackWrapper(env)
    
    # Initialize the environment
    obs, info = env.reset()
    
    episode_over = False
    total_reward = 0
    
    print("Starting Car Racing simulation")
    
    while not episode_over:
        # Sample a random action from the action space
        # action = env.action_space.sample()
        action = forward_acceleration_action()
        # Step through the environment
        obs, reward, terminated, truncated, info = env.step(action)
        # Update the cumulative reward
        total_reward += reward
        # Check if the episode is over
        episode_over = terminated or truncated

    print(f"Episode finished. Total Reward: {total_reward}")
    
    # Close the environment
    env.close()

if __name__ == "__main__":
    main()
    
    