import torch
import argparse
import yaml
import os

from box import Box
from tools.utils import print_colored, print_centered_message
from model.model import ImgEncoder  # Import ImgEncoder from main.py

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
    
    # 

if __name__ == "__main__":
    test()