import argparse
import os
import yaml
import torch
import numpy as np
import random

from box import Box
from utils.utils import print_colored, print_centered_message

from model import get_model
from engine.train import train

def load_config(config_path):
    """ Load YAML configuration file """
    with open(config_path, 'r') as file:
        cfg = yaml.safe_load(file)
    return Box(cfg)

def main():
    
    # Argument Parser
    parser = argparse.ArgumentParser(description="Training script for TransDreamer")
    parser.add_argument(
        '--config', 
        type=str, 
        default=os.path.join("config/box2d_carracing.yaml"), 
        help='path to the config file'
    )
    args = parser.parse_args()
    
    # Init
    print_colored("=" * 80, "cyan")
    print_centered_message("Initialization", " ", 80, "cyan")
    print_colored("=" * 80, "cyan")
    
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
        print_colored(f"\U00002714 Successfully loaded configuration file '{args.config}'", "dark_white")
    except Exception as e :
        print_colored(f"Error loading configuration file : {e}", "red")
        return
    print_colored("\U00002611 Done", "green")
    
    # Device
    print_colored("\U000025A2 Checking Device...", "blue")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_colored(f"\U00002714 Device : {device}", "dark_white") # [DEBUG]
    print_colored("\U00002611 Done", "green")
    
    # Initialize Model
    print_colored("\U000025A2 Initializing Model...", "blue")
    model = get_model(cfg)
    print_colored("\U00002611 Done", "green")
    
    # Set seed for reproducibility
    print_colored("\U000025A2 Setting Seed...", "blue")
    if cfg.reproducibility:
        print_colored(f"\U00002714 Seed : {cfg.seed}", "dark_white")
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed) # if use multi-GPU
        torch.backends.cudnn.deterministic = True # may have a negative impact on performance
        torch.backends.cudnn.benchmark = False # may have a positive impact on performance
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        print_colored("\U00002611 Done", "green")
    else :
        print_colored("\U00002714 Skip setting seed for reproducibility", "dark_white")
        print_colored("\U00002611 Done", "green")
    
    # Training
    print_colored("=" * 80, "cyan")
    print_centered_message("Training", " ", 80, "cyan")
    print_colored("=" * 80, "cyan")
    # TODO : Add training code here
    try:
        train(model, cfg, device, verbose=0)
        pass
    except Exception as e:
        print_colored(f"Error during training : {e}", "red")
        return

if __name__ == '__main__':
    main()