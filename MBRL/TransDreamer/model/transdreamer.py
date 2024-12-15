import torch.nn as nn

from .model import TransformerWorldModel, ActionDecoder, DenseDecoder

class TransDreamer(nn.Module):
    
    def __init__(self, cfg):
        
        super().__init__()
        
        # Configs
        # - dense input size
        self.stoch_category_size = cfg.arch.world_model.TSSM.stoch_category_size
        self.stoch_class_size = cfg.arch.world_model.TSSM.stoch_class_size
        self.d_model = cfg.arch.world_model.Transformer.d_model
        self.deter_type = cfg.arch.world_model.Transformer.deter_type
        self.num_layers = cfg.arch.world_model.Transformer.num_layers
        if self.deter_type == 'concat_all_layers':
            self.deter_size = self.num_layers * self.d_model
        self.dense_input_size = self.deter_size + self.stoch_category_size * self.stoch_class_size
        # - actor
        self.action_size = cfg.env.action_size
        self.actor_layers = cfg.arch.actor.layers
        self.actor_num_units = cfg.arch.actor.num_units
        self.actor_dist = cfg.arch.actor.dist
        # - critic
        self.critic_layers = cfg.arch.critic.layers
        self.critic_num_units = cfg.arch.critic.num_units
        
        # [REMINDER] 
        # 1) I removed all the qTransformer implementations
        # 2) I removed the case when self.stoch_discrete(renamed as self.stoch_category_size) is 0
        assert self.stoch_category_size, "Continuous latent space is not implemented yet"
            
        # TransDreamer
        self.world_model = TransformerWorldModel(cfg)
        self.actor = ActionDecoder(self.dense_input_size, self.action_size, self.actor_layers, self.actor_num_units,
                                   dist=self.actor_dist)
        self.value = DenseDecoder(self.dense_input_size, self.critic_layers, self.critic_num_units, (1,))
        self.slow_value = DenseDecoder(self.dense_input_size, self.critic_layers, self.critic_num_units, (1,))
        
    def forward(self):
        raise NotImplementedError