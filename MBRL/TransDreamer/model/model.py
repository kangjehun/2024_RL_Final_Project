import torch
import pdb

import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Independent, Normal, Bernoulli
from torch.distributions import kl_divergence
from torch.distributions.one_hot_categorical import OneHotCategorical
from collections import defaultdict

from .transformer import Transformer
from .custom import Conv2DBlock, ConvTranspose2DBlock
from .custom import Linear, MLP
from .distributions import SafeTruncatedNormal, ContinuousDist

class TransformerWorldModel(nn.Module):
    
    def __init__(self, cfg):
        
        super().__init__()
        
        # Configs
        # - train
        self.episode_length = cfg.train.episode_length
        self.log_every_step = int(float(cfg.train.log_every))
        self.log_grad = cfg.train.log_grad
        # - loss
        self.kl_scale = cfg.loss.kl_scale
        self.kl_balance = cfg.loss.kl_balance
        self.free_nats = cfg.loss.free_nats
        # - arch / world model
        self.horizon = cfg.arch.world_model.horizon
        self.pcont_layers = cfg.arch.world_model.Pcont.layers
        self.pcont_num_units = cfg.arch.world_model.Pcont.num_units
        # - arch / reward
        self.reward_layers = cfg.arch.world_model.Reward.layers
        self.reward_num_units = cfg.arch.world_model.Reward.num_units
        self.reward_transform = dict(
            tanh = torch.tanh,
            sigmoid = torch.sigmoid,
            none=nn.Identity(),
        )[cfg.arch.world_model.Reward.transform]
        # - env
        self.observation_type = cfg.env.observation.image.bev.type
        # - rl
        self.discount = cfg.rl.discount
        self.pcont_scale = cfg.loss.pcont_scale
        # - optimize
        self.grad_clip = cfg.optimize.grad_clip
        # - dense input size
        self.stoch_category_size = cfg.arch.world_model.TSSM.stoch_category_size
        self.stoch_class_size = cfg.arch.world_model.TSSM.stoch_class_size
        self.d_model = cfg.arch.world_model.Transformer.d_model
        self.deter_type = cfg.arch.world_model.Transformer.deter_type
        self.num_layers = cfg.arch.world_model.Transformer.num_layers
        if self.deter_type == 'concat_all_layers':
            self.deter_size = self.num_layers * self.d_model
        self.dense_input_size = self.deter_size + self.stoch_category_size * self.stoch_class_size
        
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
    
    def compute_loss(self, traj, global_step):
        
        self.train()
        self.requires_grad_(True)
        
        # world model rollout to obtain state representation
        prior_state, post_state = self.dynamic(traj)
        # compute world model loss given state representation
        model_loss, model_logs = self.world_model_loss(traj, global_step, prior_state, post_state)
        
        return model_loss, model_logs, prior_state, post_state

    def world_model_loss(self, traj, global_step, prior_state, post_state):
        
        # Extract and Normalize the observations
        obs = traj[self.observation_type]
        obs = obs / 255. - 0.5
        
        # Extract the rewards
        reward = traj['reward']
        reward = self.reward_transform(reward).float()
        
        post_state_trimed = {}
        for k, v in post_state.items():
            if k in ['stoch', 'logits']:
                post_state_trimed[k] = v[:, 1:] # B, 0:tau, ... -> B, 1:tau, ...
            elif k in ['deter', 'transformer_layer_outputs']:
                post_state_trimed[k] = v # B, 1:tau, ...
            else:
                raise NotImplementedError(k)
        
        transformer_feature = self.dynamic.get_feature(post_state_trimed) # B, 1:tau, d_deter+d_stoch
        # seq_len = self.H # TODO [REMINDER] I think it should be (episode length -1) = T-1
        seq_len = self.episode_length -1 
        
        image_pred_pdf = self.img_dec(transformer_feature) # B, T-1=1:tau, C, H, W
        reward_pred_pdf = self.reward(transformer_feature) # B, T-1=1:tau, 1
        pcont_pred_pmf = self.pcont(transformer_feature) # B, T-1=1:tau, 1
        
        # Compute the pcont loss
        pcont_target = self.discount * (1. - traj['terminated'][:, 1:].float()) # B, T-1=1:tau, 1
        pcont_loss = -(pcont_pred_pmf.log_prob((pcont_target.unsqueeze(2) > 0.5).float())).sum(-1) / seq_len
        pcont_loss = self.pcont_scale * pcont_loss.mean()
        pcont_accuracy = ((pcont_pred_pmf.mean == pcont_target.unsqueeze(2)).float().squeeze(-1)).sum(-1) / seq_len
        pcont_accuracy = pcont_accuracy.mean()
        
        # Compute image prediction loss
        image_pred_loss = -(image_pred_pdf.log_prob(obs[:, 1:])).sum(-1).float() / seq_len
        image_pred_loss = image_pred_loss.mean()
        # - MSE loss
        pixel_loss = F.mse_loss(image_pred_pdf.mean, obs[:, 1:], reduction='none') # B, T-1=1:tau, C, H, W
        pixel_loss = pixel_loss.flatten(start_dim=-3).sum(-1)
        mse_loss = pixel_loss.sum(-1) / seq_len
        
        # Compute reward prediction loss
        reward_pred_loss = -(reward_pred_pdf.log_prob(reward[:, 1:].unsqueeze(2))).sum(-1) / seq_len # B, T-1=1:tau, 1
        reward_pred_loss = reward_pred_loss.mean()
        predicted_reward = reward_pred_pdf.mean
        
        # Compute the KL divergence between the posterior and prior
        prior_dist = self.dynamic.get_dist(prior_state)
        post_dist = self.dynamic.get_dist(post_state_trimed)
        value_lhs = kl_divergence(post_dist, self.dynamic.get_dist(prior_state, detach=True))
        value_rhs = kl_divergence(self.dynamic.get_dist(post_state_trimed, detach=True), prior_dist)
        value_lhs = value_lhs.sum(-1) / seq_len
        value_rhs = value_rhs.sum(-1) / seq_len
        loss_lhs = torch.maximum(value_lhs.mean(), value_lhs.new_ones(value_lhs.mean().shape) * self.free_nats)
        loss_rhs = torch.maximum(value_rhs.mean(), value_rhs.new_ones(value_rhs.mean().shape) * self.free_nats)
        kl_loss = (1. - self.kl_balance) * loss_lhs + self.kl_balance * loss_rhs
        kl_loss = self.kl_scale * kl_loss
        
        # Compute the total loss
        model_loss = image_pred_loss + reward_pred_loss + pcont_loss + kl_loss
        
        if global_step % self.log_every_step == 0:
            post_dist = Independent(OneHotCategorical(logits=post_state_trimed['logits']), 1)
            prior_dist = Independent(OneHotCategorical(logits=prior_state['logits']), 1)
            logs = {
                'model_loss': model_loss.detach().item(),
                'kl_loss': kl_loss.detach().item(),
                'kl_scale': self.kl_scale,
                'reward_pref_loss': reward_pred_loss.detach().item(),
                'reward_ground_truth': reward[:, 1:].detach(),
                'reward_predicted': predicted_reward.detach().squeeze(-1),
                'image_pred_loss': image_pred_loss.detach().item(),
                'image_mse_loss': mse_loss.detach(),
                'pcont_loss': pcont_loss.detach().item(),
                'pcont_accuracy': pcont_accuracy.detach(),
                'pcont_predicted': pcont_pred_pmf.mean.detach().squeeze(-1),
                'prior_state': {k: v.detach() for k, v in prior_state.items()},
                'prior_entropy': prior_dist.entropy().mean().detach().item(),
                'post_state': {k: v.detach() for k, v in post_state.items()},
                'post_entropy': post_dist.entropy().mean().detach().item(),
            }
        else:
            logs = {}
        
        return model_loss, logs
    
    def optimize_world_model(self, model_loss, model_optimizer, writer, global_step):
        
        model_loss.backward()
        grad_norm_model = torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        if (global_step % self.log_every_step == 0) and self.log_grad:
            for name, param in self.named_parameters():
                if param.requires_grad:
                    try:
                        writer.add_scalar('grads/' + name, param.grad.norm(2), global_step)
                    except:
                        pdb.set_trace()
        model_optimizer.step()
        return grad_norm_model.item()
    
    def imagine_ahead(self, traj, post_state, actor, sample_len):
        
        self.eval()
        self.requires_grad_(False)
        
        action = traj['action']
        
        # randomly choose a state to start imagination
        # - TODO [REMINDER] Seems to assume that self.horizon is smaller than equal to episode_length
        assert self.horizon <= self.episode_length
        
        min_idx = self.horizon - 2 # TODO [REMINDER] I think -1 is enough
        perm = torch.randperm(min_idx, device=action.device)
        min_idx = perm[0] + 1
        
        pred_state = defaultdict(list)
        
        post_stoch = post_state['stoch'][:, :min_idx]
        action = action[:, :min_idx] # TODO [IMPORTANT] Why [:, 1:]? I think it should be [:, :min_idx]
        imag_transformer_feature_list = []
        imag_action_list = []
        
        # Imagination (at least two steps, recall -2)
        for _ in range(self.episode_length - min_idx):
            
            # get the prior state
            pred_prior = self.dynamic.infer_prior_stoch(post_stoch[:, -sample_len:], action[:, -sample_len:])
            transformer_feature = self.dynamic.get_feature(pred_prior)
            
            pred_action_pdf = actor(transformer_feature[:, -1:].detach()) # B, 1, d_action
            imag_action = pred_action_pdf.sample()
            imag_action = imag_action + pred_action_pdf.mean - pred_action_pdf.mean.detach() # straight through
            action = torch.cat([action, imag_action], dim=1)
            
            for k, v in pred_prior.items():
                pred_state[k].append(v[:,-1:])
            post_stoch = torch.cat([post_stoch, pred_prior['stoch'][:, -1:]], dim=1)
            
            imag_transformer_feature_list.append(transformer_feature[:, -1:])
            imag_action_list.append(imag_action)
            
        for k, v in pred_state.items():
            pred_state[k] = torch.cat(v, dim=1) 
        
        actions = torch.cat(imag_action_list, dim=1)  # B, H, d_action
        transformer_features = torch.cat(imag_transformer_feature_list, dim=1) # B, H, d_deter+d_stoch
        
        reward = self.reward(transformer_features).mean # B, H, 1
        pcont = self.pcont(transformer_features).mean # B, H, 1
        
        return pred_state, transformer_features, actions, reward, pcont, min_idx   
    
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
    
    def forward(self, traj):
        """_summary_
        Args:
            traj (dict): 
                observations (tensor): (B, 0:tau, C, H, W)
                actions (tensor): (B, 0:tau, d_action)
                dones (scalar): (B, 0:tau)
                * where tau = 0, 1, 2, 3 ...
        Returns:
            prior (dict): (B, 1:tau, ...)
            post (dict): (B, 0:tau, ...)
        """
        
        # Extract and Normalize the observations
        obs = traj[self.observation_type]
        obs = obs / 255. - 0.5
        obs_emb = self.img_enc(obs)
        
        # Extract the actions and dones
        actions = traj['action']
        # dones = traj['done'] # TODO [REMINDER] necessary?
        
        # Posterior and Prior Inference
        # z_t ~ p(z_t|x_t)
        post = self.infer_post_stoch(obs_emb)
        prev_stoch = post['stoch'][:, :-1]
        prev_action = actions[:, :-1] # TODO [REMINDER] Why [:, 1:]? I think it should be [:, :-1]
        # z^_t ~ q(z^_t|h_t) = q(z^t|Transformer(z_1:t-1, a_1:t-1))
        prior = self.infer_prior_stoch(prev_stoch, prev_action)
        
        post['deter'] = prior['deter']
        post['transformer_layer_outputs'] = prior['transformer_layer_outputs']
        # TODO [REMINDER] necessary?
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
        # TODO [REMINDER] I removed action and temp argument
        
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
        # TODO [REMINDER] I removed action and temp argument
        
        # Create embedding for the previous stochastic state and action
        B, T, N, C = prev_stoch.shape
        prev_stoch = prev_stoch.reshape(B, T, N*C)
        act_sto_emb = self.action_stoch_emb(torch.cat([prev_action, prev_stoch], dim=-1))
        act_sto_emb = F.elu(act_sto_emb)
        x = act_sto_emb.reshape(B, T, -1, 1, 1) # B, T, D, 1, 1 where D = d_model
        o = self.transformer(x) # B, T, L, D, H, W
        o = o.reshape(B, T, self.num_layers, -1) # B, T, L, D
        if self.deter_type == 'concat_all_layers':
            deter = o.reshape(B, T, -1) # B, T, L*D
        else:
            deter = o[:, :, -1] # B, T, D
            
        logits = self.prior_stoch_emb(deter).float() # B, T, N*C
        logits = logits.reshape(B, T, N, C) # B, T, N, C
        
        prior_state = self.stochastic_layer(logits)
        prior_state['deter'] = deter # B, T, L*D
        prior_state['transformer_layer_outputs'] = o # B, T, L, D
        
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
            stoch = stoch + dist.mean - dist.mean.detach() # straight through
        else:
            raise NotImplementedError
        
        state = {'logits': logits, 'stoch': stoch}
        return state
    
    def get_feature(self, state):
        """_summary_
        Args:
            state (dict): 
                stoch (distribution): batch_shape=(B, T), event_shape=(stoch_category_size, stoch_class_size) = (N, C)
                deter (tensor): (B, T, L*D) if concat_all_layers else (B, T, D)
        """
        
        shape = state['stoch'].shape
        stoch = state['stoch'].reshape([*shape[:-2]] + [self.stoch_category_size * self.stoch_class_size]) # B, T, N*C
        deter = state['deter'] # B, T, L*D
        return torch.cat([stoch, deter], dim=-1) # B, T, N*C + L*D
    
    def get_dist(self, state, detach=False):
        
        return self.get_discrete_dist(state, detach)
    
    def get_discrete_dist(self, state, detach):
        
        logits = state['logits']
        
        if detach:
            logits = logits.detach()
        
        if self.discrete_type == 'discrete':
            dist = Independent(OneHotCategorical(logits=logits), 1)
        elif self.discrete_type == 'gumbel':
            raise NotImplementedError
        
        return dist
        
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
            