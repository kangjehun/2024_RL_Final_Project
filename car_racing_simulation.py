import gymnasium as gym

def main():
    # Create the CarRacing-v3 environment
    env = gym.make(
        "CarRacing-v3",
        render_mode="human",
        domain_randomization=False,
        continuous=True,
    )
    # Initialize the environment
    obs, info = env.reset()
    
    episode_over = False
    total_reward = 0
    
    print("Starting Car Racing simulation")
    
    
    