import os
import pickle
import torch

import torch.nn as nn

class Checkpointer(object):
    
    def __init__(self, checkpoint_dir, max_num=3):
        
        # Maximum number of checkpoints to keep
        self.max_num = max_num
        # Create checkpoint directory if it doesn't exist
        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.model_list_path = os.path.join(checkpoint_dir, 'model_list.pkl')
        
        # Create model list if it doesn't exist
        if not os.path.exists(self.model_list_path):
            model_list = []
            with open(self.model_list_path, 'wb') as f:
                pickle.dump(model_list, f)
    
    def load(self, path, model_idx=1):
        
        if path == '':
            # Load the latest model with the specified index
            with open(self.model_list_path, 'rb') as f:
                model_list = pickle.load(f)
                
            if len(model_list) == 0:
                return None
            else:
                num_files = len(model_list)
                model_idx = min(num_files, model_idx)
                path = model_list[-model_idx] # Load the latest model
                checkpoint = torch.load(path)
        else:
            # Load from the specified path
            assert os.path.exists(path), f'checkpoint {path} not exists'
            checkpoint = torch.load(path)
        return checkpoint
            
    def save(self, path, model, optimizers, global_step, env_step):
        
        if path == '':
            
            path = os.path.join(self.checkpoint_dir, 'model_{:09}.pth'.format(global_step + 1))
            
            with open(self.model_list_path, 'rb+') as f:
                # Delete the oldest model if the number of models exceeds the maximum number
                model_list = pickle.load(f)
                if len(model_list) >= self.max_num:
                    if os.path.exists(model_list[0]):
                        os.remove(model_list[0])
                    del model_list[0]
                # Save the new model path to the model list
                model_list.append(path)
            
            with open(self.model_list_path, 'rb+') as f:
                # Write the new model
                pickle.dump(model_list, f)
            
            if isinstance(model, nn.DataParallel):
                model = model.module
                
            checkpoint = {
                'model': model.state_dict(),
                'global_step': global_step,
                'env_step': env_step,
            }
            
            if isinstance(optimizers, dict):
                for k, v in optimizers.items():
                    if v is not None:
                        checkpoint.update({
                            k: v.state_dict(),
                        })
            else:
                checkpoint.update({
                    'optimizer': optimizers.state_dict(),
                })
                
            assert path.endswith('.pth')
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            with open(path, 'wb') as f:
                torch.save(checkpoint, f)
            
# Functions for print
def print_colored(message, color):
    colors = {
        # Normal colors
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "purple": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        # Dark colors
        "dark_red": "\033[31m",
        "dark_green": "\033[32m",
        "dark_yellow": "\033[33m",
        "dark_blue": "\033[34m",
        "dark_purple": "\033[35m",
        "dark_cyan": "\033[36m",
        "dark_white": "\033[37m",
        # Bright colors
        "bright_red": "\033[91;1m",
        "bright_green": "\033[92;1m",
        "bright_yellow": "\033[93;1m",
        "bright_blue": "\033[94;1m",
        "bright_purple": "\033[95;1m",
        "bright_cyan": "\033[96;1m",
        "bright_white": "\033[97;1m",
        "end": "\033[0m",
    }
    print(f"{colors.get(color, colors['end'])}{message}{colors['end']}")   
    
def print_centered_message(message, border_char, total_width, color=None):
    """ Print a message centered within a border of a specific character """
    border_length = total_width
    message_length = len(message)
    padding = (border_length - message_length) // 2
    if padding > 0:
        line = border_char * padding + message + border_char * padding
        # If the total width isn't even, add an extra border character to the right
        if len(line) < total_width:
            line += border_char
    else:
        line = message
    if color is not None:
        print_colored(line, color)
    else:
        print(line)    
    
