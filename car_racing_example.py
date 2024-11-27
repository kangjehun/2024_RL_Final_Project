import gymnasium as gym

# Select 

def main():
    # Create the CarRacing-v2 environment
    env = gym.make(
        "CarRacing-v2",
        render_mode="human",
        domain_randomize=False,
        continuous=True,
    )
    # Initialize the environment
    obs, info = env.reset()
    
    episode_over = False
    total_reward = 0
    
    print("Starting Car Racing simulation")
    
    while not episode_over:
        # Sample a random action from the action space
        action = env.action_space.sample()
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
    
    