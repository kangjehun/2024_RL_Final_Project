import os
import argparse

from stable_baselines3 import SAC, PPO, DDPG
from stable_baselines3.common.vec_env import VecVideoRecorder
from env.env_utils import create_carracing_env

class RLTester:
    """ A class to test or record videos of trained RL agents """
    
    # Supported Algorithms
    ALGORITHMS = {
        "SAC": SAC,
        "PPO": PPO,
        "DDPG": DDPG,
    }
    
    def __init__(self, algorithm, video_length=1000, record=False):
        """
        Initialize the VideoRecorder
        
        Args:
            video_length (int): The length of the video in timesteps
            record (bool): Whether to record the video or not
        """
        
        # Check if the algorithm is supported
        if algorithm not in self.ALGORITHMS:
            raise ValueError(f"Algorithm not supported. Supported algorithms are: {self.ALGORITHMS.keys()}")
        
        # Initialize the VideoRecorder
        self.algorithm = algorithm
        self.video_length = video_length
        self.record = record
        
        print(f"video_length: {self.video_length}")
        
        # Paths
        self.video_folder = f"Baselines3/{self.algorithm}/video_record/video"
        self.model_path = f"Baselines3/{self.algorithm}/model/best_model.zip"
        
        # Ensure the video folder exists if recording
        if self.record:
            os.makedirs(self.video_folder, exist_ok=True)
        
        # Environment setup
        self.eval_env = self._create_environment()
        
        # Load the trained model
        self.model = self._load_model()
        
    def _create_environment(self):
        """ Create the evaluation environment and wrap it for video recording """
        render_mode = "rgb_array" if self.record else "human"
        env = create_carracing_env(render_mode=render_mode, use_subproc=False, num_envs=1, train=False, max_episode_steps=self.video_length)
        if self.record:
            env = VecVideoRecorder(
                env,
                self.video_folder,
                record_video_trigger=lambda x: x == 0, # Record the first episode only
                video_length=self.video_length,
                name_prefix=f"{self.algorithm.lower()}-test"
            )
        return env

    def _load_model(self):
        """ Load the trained model """
        return self.ALGORITHMS[self.algorithm].load(self.model_path)
    
    def run(self):
        """ Run the trained agent in the environment """
        
        # Log
        if self.record:
            print(f"Recording video of {self.algorithm} agent playing...")
        else:
            print(f"Running {self.algorithm} agent in the carracing environment...")
        
        # Reset the environment
        obs = self.eval_env.reset()
        
        # Run the agent in the environment
        timestamp = 0
        for _ in range(self.video_length):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, done , _ = self.eval_env.step(action)
            timestamp += 1
            print(f"Timestamp: {timestamp}")
            if not self.record:
                self.eval_env.render()
            if done:
                break
        
        # Close the environment
        self.eval_env.close()
        print(f"Environment closed after {timestamp} timesteps")
        if self.record:
            print(f"Video recording for {self.algorithm} agent completed. Saved in {self.video_folder}")
        else:
            print(f"Testing of {self.algorithm} agent completed")

def parse_args():
    """ Parse command-line arguments """
    
    parser = argparse.ArgumentParser(description="Record videos of trained RL agents.")
    parser.add_argument(
        "--algorithm",
        type=str,
        required=True,
        choices=RLTester.ALGORITHMS.keys(),
        help="The RL algorithm to use (SAC, PPO, DDPG)."
    )
    parser.add_argument(
        "--video_length",
        type=int,
        default=1000,
        help="Length of the video in timesteps (default: 1000)."
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="If set, records a video; otherwise, runs a normal test."
    )
    return parser.parse_args()

if __name__ == "__main__":
    
    args = parse_args()
    
    tester = RLTester(algorithm=args.algorithm, video_length=args.video_length, record=args.record)
    tester.run()