import torch
import argparse
import yaml
import os

from box import Box
from utils.utils import print_colored, print_centered_message
from model.model import ImgEncoder  # Import ImgEncoder from main.py

# distribution
from torch.distributions import OneHotCategorical, Independent

def load_config(config_path):
    """ Load YAML configuration file """
    with open(config_path, 'r') as file:
        cfg = yaml.safe_load(file)
    return Box(cfg)

# Test function
def test():

    # Argument Parser
    parser = argparse.ArgumentParser(description="Training script for TransDreamer")
    parser.add_argument(
        '--config', 
        type=str, 
        default=os.path.join("config/box2d_carracing.yaml"), 
        help='path to the config file'
    )
    args = parser.parse_args()
    
    # Configuration
    print_colored("\U000025A2 Configuration...", "blue")
    # - check if the configuration file exists
    abs_config_path = os.path.abspath(args.config)
    if not os.path.exists(abs_config_path):
        print_colored(f"Config file '{args.config}' does not exit.", "red")
        return
    # - load the configuration file
    try:
        cfg = load_config(abs_config_path)
        print_colored(f"Successfully loaded configuration file '{args.config}'", "dark_white")
    except Exception as e :
        print_colored(f"Error loading configuration file : {e}", "red")
        return
    print_colored("\U00002611 Done", "green")

    #################### Test ####################
    
    # ImgEncoder Test
    print_colored("\U000025A2 Testing ImgEncoder...", "blue")
    # - initialize the ImgEncoder
    encoder = ImgEncoder(cfg)
    # - print the shapes
    print(f"Flattened feature dimension: {encoder.final_dim}") # Flattened feature dimension
    print_colored("\U00002611 Done", "green")
    
    # Distribution Test
    # print_colored("\U000025A2 Testing Distribution...", "blue")
    # # - dummy logits
    # B, T, stoch_category_size, stoch_class_size = 2, 3, 4, 5
    # logits = torch.randn(B, T, stoch_category_size, stoch_class_size)
    # # - without independent
    # dist_without_independent = OneHotCategorical(logits=logits)
    # sample_without = dist_without_independent.sample()
    # log_probs_without = dist_without_independent.log_prob(sample_without)
    # # - with independent
    # dist_with_independent = Independent(dist_without_independent, 1)
    # sample_with = dist_with_independent.sample()
    # log_probs_with = dist_with_independent.log_prob(sample_with)
    # # - print the shapes
    # print(f"- Without Independent:")
    # print("Sample Shape: ", sample_without.shape)
    # print("Batch Shape: ", dist_without_independent.batch_shape)
    # print("Event Shape: ", dist_without_independent.event_shape)
    # print(sample_without)
    # print("Log-Probabilties Shape: ", log_probs_without.shape)
    # print(log_probs_without)
    # print(f"- With Independent:")
    # print("Sample Shape: ", sample_with.shape)
    # print("Batch Shape: ", dist_with_independent.batch_shape)
    # print("Event Shape: ", dist_with_independent.event_shape)
    # print(sample_with)
    # print("Log-Probabilties Shape: ", log_probs_with.shape)
    # print(log_probs_with)

if __name__ == "__main__":
    test()