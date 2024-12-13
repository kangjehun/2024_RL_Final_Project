import torch.nn as nn

from .model import TransformerWorldModel

class TransDreamer(nn.Module):
    
    def __init__(self, cfg):
        
        super().__init__()
        self.world_model = TransformerWorldModel(cfg)