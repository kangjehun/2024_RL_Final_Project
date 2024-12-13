import argparse
import os

from env.env_utils import create_carracing_env
from stable_baselines3 import SAC, PPO, DDPG
from stable_baselines3.common.callbacks import EvalCallback

class RLTrainer:
    
    # Supported algorithms
    ALGORITHMS = {
        "SAC": SAC,
        "PPO": PPO,
        "DDPG": DDPG,
    }
    
    def __init__(self, algorithm, timesteps, render_mode, use_subproc, num_envs):
        
        # Initialize the RL Trainer
        self.algorithm = algorithm
        self.timesteps = timesteps
        self.render_mode = render_mode
        self.use_subproc = use_subproc
        self.num_envs = num_envs
        
        # Paths
        self.project_root = os.path.abspath(os.path.dirname(__file__))
        self.model_dir = os.path.join(self.project_root, f"Baselines3/{self.algorithm.upper()}/model/")
        self.tensorboard_log_dir = os.path.join(self.project_root, f"Baselines3/{self.algorithm.upper()}/tensorboard/")
        self.eval_log_path = os.path.join(self.project_root, f"Baselines3/{self.algorithm.upper()}/eval/eval_logs/")
        self.best_model_dir = self.model_dir
        self.last_model_dir = os.path.join(self.model_dir, "last_model.zip")
        
        # Create necessary directories
        self._create_directories()
    
    def _create_directories(self):
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.tensorboard_log_dir, exist_ok=True)
        os.makedirs(self.eval_log_path, exist_ok=True)
    
    def _create_environment(self):
        # Initialize the training environment
        return create_carracing_env(
            render_mode = self.render_mode,
            use_subproc=self.use_subproc,
            num_envs=self.num_envs,
        )
        
    def _initialize_model(self, env):
        # Get the selected RL algorithm class
        AlgorithmClass = self.ALGORITHMS[self.algorithm]
        
        # Common arguments for all algorithms
        common_args = {
            "policy": "CnnPolicy",
            "env": env,
            "verbose": 2,
            "tensorboard_log": self.tensorboard_log_dir,
        }
        
        # Algorithm specific arguments
        if self.algorithm == "SAC":
            specific_args = {
                "buffer_size": 100000,
                "learning_rate": 3e-4,
                "batch_size": 256,
                "train_freq": 1,
                "gradient_steps": 1,
                "learning_starts": 1000,
                "gamma": 0.99,
                "tau": 0.005,
                "ent_coef": "auto",
            }
        elif self.algorithm == "PPO":
            specific_args = {
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
            }
        elif self.algorithm == "DDPG":
            specific_args = {
                "buffer_size": 100000,
                "learning_rate": 1e-3,
                "batch_size": 64,
                "train_freq": (1, "episode"),
                "gradient_steps": -1,
                "learning_starts": 1000,
                "gamma": 0.99,
                "tau": 0.005,
            }
        else :
            raise ValueError(f"Unsupported Algorithm: {self.algorithm}")
        
        # Combine common and specific arguments
        model_args = {**common_args, **specific_args}
        
        # Initialize the RL model
        return AlgorithmClass(**model_args)
    
    def _get_eval_callback(self, env):
        # Create an evaluation callback
        return EvalCallback(
            env,
            best_model_save_path = self.best_model_dir,
            log_path = self.eval_log_path,
            eval_freq = 10000,
            n_eval_episodes = 5,
            deterministic = True,
            render = True
        )
    
    def train(self):
        
        # Create the training environment
        env = self._create_environment()
        
        # Initialize the RL model
        model = self._initialize_model(env)
        
        # Set up evaluation callback
        eval_callback = self._get_eval_callback(env)
        
        # Train the model
        print(f"Training {self.algorithm.upper()} for {self.timesteps} timesteps ...")
        model.learn(total_timesteps=self.timesteps, callback=eval_callback)
        
        # Save the final model
        print(f"Saving model to {self.last_model_dir}...")
        model.save(self.last_model_dir)
        print("Training completed!")        
        
        
        
def parse_ars():
    # Argument parser for command-line options
    parser = argparse.ArgumentParser(description="Train a Reinforcement Learning Agent")
    parser.add_argument(
        "--algorithm",
        type=str,
        required=True,
        choices=RLTrainer.ALGORITHMS.keys(),
        help="The RL Algorithm to use (SAC, PPO, DDPG)"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        required=True,
        default=100000,
        help="Total number of timestaps for training"
    )
    parser.add_argument(
        "--render_mode",
        type=str,
        default="rgb_array",
        choices=["rgb_array", "human"],
        help="Rendering mode for the environment"
    )
    parser.add_argument(
        "--use_subproc",
        action="store_true",
        help="Use Subprocesses for parallel environment"
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help="Number of environments for parallelization"
    )
    return parser.parse_args()

if __name__ == "__main__":
    # Parse commnd-line arguments
    args = parse_ars()
    
    # Create an instance of RLTrainer
    trainer = RLTrainer(
        algorithm=args.algorithm,
        timesteps=args.timesteps,
        render_mode=args.render_mode,
        use_subproc=args.use_subproc,
        num_envs=args.num_envs,
    )
    
    # Start Training
    trainer.train()