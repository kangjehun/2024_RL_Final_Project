import torch
import torch.nn as nn

from .transformer import Transformer
from .custom import Conv2DBlock, Linear, MLP

class TransformerWorldModel(nn.Module):
    
    def __init__(self, cfg):
        
        super().__init__()
        
        self.dynamic = TransformerDynamic(cfg)

class TransformerDynamic(nn.Module):
    
    def __init__(self, cfg):
        
        super().__init__()
        
        # Config
        self.action_size = cfg.env.action_size
        self.weight_init = cfg.arch.world_model.TSSM.weight_init
        self.stoch_size = cfg.arch.world_model.TSSM.stoch_size
        self.stoch_discrete = cfg.arch.world_model.TSSM.stoch_discrete
        self.d_model = cfg.arch.world_model.Transformer.d_model
        
        self.img_enc = ImgEncoder(cfg)
        self.img_feature_dim = self.img_enc.final_dim # dimension of z
        self.cell = Transformer(cfg.arch.world_model.transformer)
        
        # TODO [REMINDER] Remove the case when self.stoch_discrete is 0
        if self.stoch_discrete:
            latent_dim = self.stoch_size * self.stoch_discrete
        else:
            raise NotImplementedError("Continuous latent space is not implemented yet")
        
        # TODO [REMINDER] Remove all the qTransformer implementations
        self.action_stoch_emb = Linear(self.action_size + latent_dim, self.d_model, weight_init=self.weight_init)
        self.post_stoch_emb = MLP()

class ImgEncoder(nn.Module):
    
    def __init__(self, cfg):
        
        super().__init__()
        
        # Config
        self.in_channels = 1 if cfg.env.grayscale else 3
        self.num_channels = cfg.arch.world_model.ImageEncoder.num_channels     
        self.kernel_size = cfg.arch.world_model.ImageEncoder.kernel_size
        self.stride = cfg.arch.world_model.ImageEncoder.stride
        self.padding = cfg.arch.world_model.ImageEncoder.padding
        self.weight_init = cfg.arch.world_model.ImageEncoder.weight_init
        
        # Input shape
        self.input_height = cfg.env.observation.height
        self.input_width = cfg.env.observation.width
        self.input_channels = cfg.env.observation.channels
        
        # Build the convolutional encoder
        self.enc = nn.Sequential(
            Conv2DBlock(self.in_channels, self.num_channels,
                        self.kernel_size, self.stride, self.padding, non_linearity=True, weight_init=self.weight_init),
            Conv2DBlock(self.num_channels, 2*self.num_channels,
                        self.kernel_size, self.stride, self.padding, non_linearity=True, weight_init=self.weight_init),
            Conv2DBlock(2*self.num_channels, 4*self.num_channels,
                        self.kernel_size, self.stride, self.padding, non_linearity=True, weight_init=self.weight_init),
            Conv2DBlock(4*self.num_channels, 8*self.num_channels,
                        self.kernel_size, self.stride, self.padding, non_linearity=False, weight_init=self.weight_init),
        )
        self.final_dim = self._compute_final_dim(cfg)
        
    def _compute_final_dim(self, cfg):
        """ Computes the final flattened output dimension based on the input shape """
        with torch.no_grad():
            c = cfg.env.observation.channels
            h = cfg.env.observation.height
            w = cfg.env.observation.width
            dummy_input = torch.zeros(1, c, h, w)
            output = self.enc(dummy_input)
            return output.numel()
    
    def forward(self, inputs):
        """_summary_

        Args:
            inputs (tensor): (B, T, C, H, W)
        Returns:
            tensor: (B, T, F)
        """
        
        shapes = inputs.size()
        
        # preprocessing
        x = inputs.view(-1, shapes[-3], shapes[-2], shapes[-1]) # (BxT, C, H, W)
        
        # encoding
        o = self.enc(x) 
        
        # Flatten the output
        o = o.view(shapes[0], shapes[1], self.final_dim)
        
        return o