import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Independent, Normal, Bernoulli
from torch.distributions.one_hot_categorical import OneHotCategorical
from .transformer import Transformer
from .custom import Conv2DBlock, ConvTranspose2DBlock
from .custom import Linear, MLP
from .distributions import SafeTruncatedNormal, ContinuousDist

class TransformerWorldModel(nn.Module):
    
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
        # - reward
        self.reward_layers = cfg.arch.world_model.Reward.layers
        self.reward_num_units = cfg.arch.world_model.Reward.num_units
        # - pcont
        self.pcont_layers = cfg.arch.world_model.Pcont.layers
        self.pcont_num_units = cfg.arch.world_model.Pcont.num_units
        
        # World Model Components
        # TSSM
        self.dynamic = TransformerDynamic(cfg)
        # Image Predictor : x^_t ~ q(x^_t|h_t, z_t)
        self.img_dec = ImgDecoder(cfg, self.dense_input_size)
        # Reward Predictor : r^_t ~ q(r^_t|h_t, z_t)
        self.reward = DenseDecoder(self.dense_input_size, self.reward_layers, self.reward_num_units, (1,))
        # Done Predictor : d^_t ~ q(d^_t|h_t, z_t)
        self.pcont = DenseDecoder(self.dense_input_size, self.pcont_layers, self.pcont_num_units, (1,), dist='binary')

    def forward(self, traj):
        # TODO [REMINDER] necessary?
        raise NotImplementedError
    
class TransformerDynamic(nn.Module):
    
    def __init__(self, cfg):
        
        super().__init__()
        
        # Config
        # - env
        self.action_size = cfg.env.action_size
        self.observation_type = cfg.env.observation.image.bev.type
        # - TSSM
        self.hidden_size = cfg.arch.world_model.TSSM.hidden_size
        self.stoch_category_size = cfg.arch.world_model.TSSM.stoch_category_size
        self.stoch_class_size = cfg.arch.world_model.TSSM.stoch_class_size
        self.latent_size = self.stoch_category_size * self.stoch_class_size
        self.discrete_type = cfg.arch.world_model.TSSM.discrete_type
        # - Transformer
        self.d_model = cfg.arch.world_model.Transformer.d_model
        self.num_layers = cfg.arch.world_model.Transformer.num_layers
        self.deter_type = cfg.arch.world_model.Transformer.deter_type
        self.deter_size = self.d_model
        if self.deter_type == 'concat_all_layers':
            self.deter_size = self.deter_size * self.num_layers
        
        self.img_enc = ImgEncoder(cfg)
        self.img_feature_dim = self.img_enc.final_dim # dimension of z
        self.transformer = Transformer(cfg.arch.world_model.Transformer)
        
        # Embeddings
        # z (stochastic state), a (action)-> h (deterministic state)
        self.action_stoch_emb = Linear(self.latent_size + self.action_size, self.d_model)
        # x'(image feature) -> z (stochastic state)
        self.post_stoch_emb = MLP([self.img_feature_dim, self.hidden_size, self.latent_size])
        # h (determinisitic state) -> z (stochastic state)
        self.prior_stoch_emb = MLP([self.deter_size, self.hidden_size, self.latent_size])
    
    def forward(self, episode):
        """_summary_
        Args:
            episode (dict): 
                observations (tensor): (B, T, C, H, W)
                actions (tensor): (B, T, d_action)
                dones (scalar): (B, T)
        Returns:
            prior (dict): 
            post (dict):
        """
        
        # Extract and Normalize the observations
        obs = episode[self.observation_type]
        obs = obs / 255. - 0.5
        obs_emb = self.img_enc(obs)
        
        # Extract the actions and dones
        actions = episode['action']
        dones = episode['done']
        
        # Posterior and Prior Inference
        # z_t ~ p(z_t|x_t)
        post = self.infer_post_stoch(obs_emb)
        prev_stoch = post['stoch'][:, :-1]
        prev_action = actions[:, :-1] # TODO [REMINDER] Why [:, 1:]? I revised it as [:, :-1]
        # z^_t ~ q(z^_t|h_t) = q(z^t|Transformer(z_1:t-1, a_1:t-1))
        prior = self.infer_prior_stoch(prev_stoch, prev_action)
        
        # TODO [REMINDER] necessary?
        # post['deter'] = prior['deter'] 
        # post['transformer_layer_outputs'] = prior['transformer_layer_outputs']
        # prior['stoch_int'] = prior['stoch'].argmax(-1).float()
        # post['stoch_int'] = post['stoch'].argmax(-1).float()
        return prior, post
    
    def infer_post_stoch(self, embeded_observation):
        """_summary_
        Args:
            embeded_observation (tensor): (B, T, F)
        Returns:
            post_state (dict): 
                logits (tensor): (B, T, stoch_category_size, stoch_class_size)
                stoch (distribution): batch_shape=(B, T), event_shape=(stoch_category_size, stoch_class_size)
        """
        # TODO [REMINDER] I removed action argument
        
        B, T, _ = embeded_observation.shape
        logits = self.post_stoch_emb(embeded_observation).float()
        logits = logits.reshape(B, T, self.stoch_category_size, self.stoch_class_size).float()
        post_state = self.stochastic_layer(logits)
        
        return post_state
        
        
    def infer_prior_stoch(self, prev_stoch, prev_action):
        """_summary_
        Args:
            prev_stoch (tensor): (B, T, stoch_category_size, stoch_class_size)
            prev_action (tensor): (B, T, d_action)
        Returns:
            prior_state (dict): 
                logits (tensor): (B, T, stoch_category_size, stoch_class_size)
                stoch (distribution): batch_shape=(B, T), event_shape=(stoch_category_size, stoch_class_size)
                deter (tensor): (B, T, L*D) if concat_all_layers else (B, T, D)
        """
        
        # Create embedding for the previous stochastic state and action
        B, T, N, C = prev_stoch.shape
        prev_stoch = prev_stoch.reshape(B, T, N*C)
        act_sto_emb = self.action_stoch_emb(torch.cat([prev_action, prev_stoch], dim=-1))
        act_sto_emb = F.elu(act_sto_emb)
        x = act_sto_emb.reshape(B, T, -1, 1, 1) # B, T, D, 1, 1 where D = d_model
        o = self.transformer(x)
        o = o.reshape(B, T, self.n_layers, -1) # B, T, L, D
        if self.deter_type == 'concat_all_layers':
            deter = o.reshape(B, T, -1)
        else:
            deter = o[:, :, -1]
        logits = self.prior_stoch_emb(deter).float() # B, T, N*C
        B, T, N, C = prev_stoch.shape
        logits = logits.reshape(B, T, N, C) # B, T, N, C
        prior_state = self.stochastic_layer(logits)
        prior_state['deter'] = deter
        # prior_state['transformer_layer_outputs'] = o # TODO [REMINDER] necessary?
        
        return prior_state
    
    def stochastic_layer(self, logits):
        """_summary_
        Args:
            logits (tensor): (B, T, stoch_category_size, stoch_class_size)
        Returns:
            state (dict): 
                logits (tensor): (B, T, stoch_category_size, stoch_class_size)
                stoch (distribution): batch_shape=(B, T), event_shape=(stoch_category_size, stoch_class_size)
        """
        
        if self.discrete_type == 'discrete':
            dist = Independent(OneHotCategorical(logits=logits), 1)     
            stoch = dist.sample()
            stoch = stoch + dist.mean - dist.mean.detach()
        else:
            raise NotImplementedError
        
        state = {'logits': logits, 'stoch': stoch}
        return state

class ImgEncoder(nn.Module):
    
    def __init__(self, cfg):
        
        super().__init__()
        
        # Config
        self.in_channels = 1 if cfg.env.grayscale else 3
        self.num_channels = cfg.arch.world_model.ImageEncoder.num_channels     
        self.kernel_size = cfg.arch.world_model.ImageEncoder.kernel_size
        self.stride = cfg.arch.world_model.ImageEncoder.stride
        self.padding = cfg.arch.world_model.ImageEncoder.padding
        
        # Input shape
        self.input_height = cfg.env.observation.image.bev.height
        self.input_width = cfg.env.observation.image.bev.width
        self.input_channels = cfg.env.observation.image.bev.channels
        
        # Build the convolutional encoder
        self.enc = nn.Sequential(
            Conv2DBlock(self.in_channels, self.num_channels, self.kernel_size, self.stride, self.padding),
            Conv2DBlock(self.num_channels, 2*self.num_channels, self.kernel_size, self.stride, self.padding),
            Conv2DBlock(2*self.num_channels, 4*self.num_channels, self.kernel_size, self.stride, self.padding),
            Conv2DBlock(4*self.num_channels, 8*self.num_channels, self.kernel_size, self.stride, self.padding, 
                        non_linearity=False),
        )
        self.final_dim = self._compute_final_dim(cfg)
        
    def _compute_final_dim(self, cfg):
        """ Computes the final flattened output dimension based on the input shape """
        with torch.no_grad():
            c = self.input_channels
            h = self.input_height
            w = self.input_width
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

class ImgDecoder(nn.Module):
    
    def __init__(self, cfg, input_size):
        
        super().__init__()
        
        self.num_channels = cfg.arch.world_model.ImageDecoder.num_channels
        self.channels = cfg.env.observation.image.bev.channels
        self.height = cfg.env.observation.image.bev.height
        self.width = cfg.env.observation.image.bev.width
        
        self.fc = Linear(input_size, self.num_channels*32)
        self.dec = nn.Sequential(
            ConvTranspose2DBlock(32*self.num_channels, 4*self.num_channels, 5, 2, 0),
            ConvTranspose2DBlock(4*self.num_channels, 2*self.num_channels, 5, 2, 0),
            ConvTranspose2DBlock(2*self.num_channels, self.num_channels, 6, 2, 0),
            ConvTranspose2DBlock(self.num_channels, self.channels, 9, 3, 0),   
        )
        self.shape = (self.channels, self.height, self.width)
        self.reconstruction_sigma = cfg.arch.world_model.ImageDecoder.reconstruction_sigma
    
    def forward(self, inputs):
        """_summary_
        Args:
            inputs (tensor): (B, T, d_model(dim of h) + latent_size(dim of z))
        Returns:
            dec_pdf (distribution): batch_shape=(B, T), event_shape=(C, H, W)
        """
        
        input_shape = inputs.shape
        fc_output = self.fc(inputs)
        dec_output = self.dec(fc_output.reshape(input_shape[0]*input_shape[1], -1, 1, 1))
        dec_output = dec_output.reshape([*input_shape[:2]] + list(self.shape))
        
        dec_pdf = Independent(Normal(dec_output, self.reconstruction_sigma * dec_output.new_ones(dec_output.shape)), 
                                     len(self.shape))

        return dec_pdf
    
    
class DenseDecoder(nn.Module):
    
    def __init__(self, input_size, num_layers, units, output_shape, weight_init='xavier', dist='normal', act='elu'):
        
        super().__init__()
        
        self.dist = dist
        self.output_shape = output_shape
        
        layers = []
        for i in range(num_layers):
            if i == 0:
                dim_in = input_size
            else:
                dim_in = units
            dim_out = units
            layers.append(Linear(dim_in, dim_out, weight_init=weight_init))
            if act == 'relu':
                layers.append(nn.ReLU())
            elif act == 'elu':
                layers.append(nn.ELU())
            else:
                layers.append(nn.CELU())
        layers.append(Linear(dim_out, 1, weight_init=weight_init))
        self.dec = nn.Sequential(*layers)
    
    def forward(self, inputs):
        
        logits = self.dec(inputs).float()
        
        if self.dist == 'normal':
            pdf = Independent(Normal(logits, 1), len(self.output_shape))
        elif self.dist == 'binary':
            pdf = Independent(Bernoulli(logits=logits), len(self.output_shape))
        else:
            raise NotImplementedError(self.dist)

        return pdf
    
class ActionDecoder(nn.Module):
    
    def __init__(self, input_size, action_size, num_layers, units, dist='onehot', min_std=0.1, 
                 act='elu', weight_init='xavier'):
        
        super().__init__()
        
        self.dist = dist
        self.min_std = min_std
        
        layers = []
        
        for i in range(num_layers):
            
            if i == 0:
                dim_in = input_size
            else:
                dim_in = units
            dim_out = units
            
            layers.append(Linear(dim_in, dim_out, weight_init=weight_init))
            if act == 'relu':
                layers.append(nn.ReLU())
            elif act == 'elu':
                layers.append(nn.ELU())
            else:
                layers.append(nn.CELU())
        
        if dist == 'onehot':
            layers.append(Linear(dim_out, action_size, weight_init=weight_init))
        elif dist == 'trunc_normal':
            layers.append(Linear(dim_out, 2*action_size, weight_init=weight_init))
        else:
            raise NotImplementedError(dist)

        self.dec = nn.Sequential(*layers)
    
    def forward(self, inputs):
        
        logits = self.dec(inputs).float()
        
        if self.dist == 'onehot':    
            dist = OneHotCategorical(logits=logits)
        elif self.dist == 'truc_normal':
            mean, std = torch.chunk(logits, 2, -1)
            mean = torch.tanh(mean)
            std = 2 * torch.sigmoid(std / 2) + self.min_std
            dist = SafeTruncatedNormal(mean, std, -1, 1)
            dist = ContinuousDist(Independent(dist, 1))
        else:
            raise NotImplementedError(self.dist)
        return dist
            