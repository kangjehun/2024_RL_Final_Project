import torch
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
    
    def policy(self, prev_obs, obs, action, state=None, training=True, episode_length=50):
        """
        prev_obs : B, C, H, W @ tau-1
        obs : B, C, H, W @ tau
        """
        obs = obs.unsqueeze(1) # B, T, C, H, W
        obs_emb = self.world_model.dynamic.img_encoder(obs) # B, T, F
        post = self.world_model.dynamic.infer_post_stoch(obs_emb)
        
        if state is None:
            state = post
            prev_obs = prev_obs.unsqueeze(1) / 255. - 0.5 # B, T, C, H, W
            prev_obs_emb = self.world_model.dynamic.img_encoder(prev_obs)
            prev_post = self.world_model.dynamic.infer_post_stoch(prev_obs_emb)
            
            for k, v in post.items():
                state[k] = torch.cat([prev_post[k], v], dim=1)
            stoch = state['stoch'] # B, 0:tau=1, d_stoch=stoch_category_size*stoch_class_size
        else:
            stoch = torch.cat([state['stoch'], post['stoch'][:, -1:]], dim=1)[:, -episode_length:] # B, 0:tau, d_stoch
            for k, v in post.items():
                state[k] = torch.cat([state[k], v], dim=1)[:, -episode_length:]
        
        # stoch[:, :-1] = B, 0:tau-1, (d_deter, d_stoch, ...)
        # pred_prior    = B, 1:tau,   (d_deter, d_stoch, ...)
        pred_prior = self.world_model.dynamic.infer_prior_stoch(stoch[:, :-1]) 
        
        post_state_trimed = {} # B, 1:tau, (d_deter, d_stoch, ...)
        for k, v in state.items():
            if k not in ['stoch', 'logits']: # TODO [REMINDER] Other keys are needed?
                raise NotImplementedError
            post_state_trimed[k] = v[:, 1:]
        post_state_trimed['deter'] = pred_prior['deter']
        post_state_trimed['transformer_layer_outputs'] = pred_prior['transformer_layer_outputs']
        
        transformer_feature = self.world_model.dynamic.get_feature(post_state_trimed) # B, 1:tau, d_deter+d_stoch
        pred_action_pdf = self.actor(transformer_feature[:, -1:]) # B, 1(tau), d_action
        if training:
            pred_action = pred_action_pdf.sample() # B, 1(tau), A=d_action
        else:
            if self.actor_dist == 'onehot':
                pred_action = pred_action_pdf.mean
                index = pred_action.argmax(dim=-1)[0]
                pred_action = torch.zeros_like(pred_action)
                pred_action[..., index] = 1
            else:
                raise NotImplementedError
        
        assert episode_length > 1 # prevent too short episode
        action = torch.cat([action, pred_action], dim=1)[:, -(episode_length-1):] # B, 1:tau-1, 
        
        return action, state
        