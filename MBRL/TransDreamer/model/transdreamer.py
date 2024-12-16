import torch
import pdb
import torch.nn as nn

from torch.distributions import Independent, OneHotCategorical
from .model import TransformerWorldModel, ActionDecoder, DenseDecoder

class TransDreamer(nn.Module):
    
    def __init__(self, cfg):
        
        super().__init__()
        
        # Configs
        # - train
        self.log_every = float(cfg.train.log_every)
        self.episode_length = cfg.train.episode_length
        self.log_grad = cfg.train.log_grad
        # - loss
        self.entropy_scale = float(cfg.loss.entropy_scale)
        # - arch / actor
        self.action_size = cfg.env.action_size
        self.actor_layers = cfg.arch.actor.layers
        self.actor_num_units = cfg.arch.actor.num_units
        self.actor_dist = cfg.arch.actor.dist
        self.actor_loss_type = cfg.arch.actor.loss_type
        # - arch / critic
        self.critic_layers = cfg.arch.critic.layers
        self.critic_num_units = cfg.arch.critic.num_units
        self.slow_update = 0
        self.slow_update_every_step = cfg.arch.critic.slow_update_every
        # - rl
        self.lambda_ = cfg.rl.lambda_
        # - dense input size
        self.stoch_category_size = cfg.arch.world_model.TSSM.stoch_category_size
        self.stoch_class_size = cfg.arch.world_model.TSSM.stoch_class_size
        self.d_model = cfg.arch.world_model.Transformer.d_model
        self.deter_type = cfg.arch.world_model.Transformer.deter_type
        self.num_layers = cfg.arch.world_model.Transformer.num_layers
        if self.deter_type == 'concat_all_layers':
            self.deter_size = self.num_layers * self.d_model
        self.dense_input_size = self.deter_size + self.stoch_category_size * self.stoch_class_size
        # - optimize
        self.grad_clip = cfg.optimize.grad_clip
        
        
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
        obs = obs.unsqueeze(1) / 255. - 0.5 # B, T, C, H, W
        obs_emb = self.world_model.dynamic.img_enc(obs) # B, T, F
        post = self.world_model.dynamic.infer_post_stoch(obs_emb)
        if state is None:
            state = post
            prev_obs = prev_obs.unsqueeze(1) / 255. - 0.5 # B, T, C, H, W
            prev_obs_emb = self.world_model.dynamic.img_enc(prev_obs)
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
        pred_prior = self.world_model.dynamic.infer_prior_stoch(stoch[:, :-1], action) 
        
        post_state_trimed = {} # B, 1:tau, (d_deter, d_stoch, ...)
        for k, v in state.items():
            if k in ['stoch', 'logits']: # TODO [REMINDER] Other keys are needed?
                post_state_trimed[k] = v[:, 1:]
            else:
                raise NotImplementedError(k)
        post_state_trimed['deter'] = pred_prior['deter']
        post_state_trimed['transformer_layer_outputs'] = pred_prior['transformer_layer_outputs']
        
        transformer_feature = self.world_model.dynamic.get_feature(post_state_trimed) # B, 1:tau, d_deter+d_stoch
        pred_action_pdf = self.actor(transformer_feature[:, -1:].detach()) # B, 1(tau), d_action
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
    
    def world_model_loss(self, traj, global_step):
        return self.world_model.compute_loss(traj, global_step)
    
    def actor_and_critic_loss(self, traj, post_state, global_step):
        
        # slow update for value network
        self.update_slow_target()
        
        # Actor Loss
        # - freeze value network
        self.value.eval()
        self.value.requires_grad_(False)
        # - imagine ahead
        imagine_state, imagine_feature, imagine_action, imagine_reward, imagine_pcont, imagine_idx = \
            self.world_model.imagine_ahead(traj, post_state, self.actor, self.episode_length-1) # B, H, ...
        # - compute target : (B, H-1, 1), (B, H, 1)
        target, weights = self.compute_target(imagine_feature, imagine_reward, imagine_pcont)
        # - compute actor loss
        slice_idx = -1
        actor_dist = self.actor(imagine_feature.detach()) # B, H, A
        if self.actor_dist == 'onehot':
            indices = imagine_action.max(-1)[1]
            actor_logprob = actor_dist._categorical.log_prob(indices) # B, H
        else:      
            raise NotImplementedError
        if self.actor_loss_type == 'reinforce':
            baseline = self.value(imagine_feature[:, :slice_idx]).mean # H-1, 1
            advantage = (target - baseline).detach() # B, H-1, 1
            actor_loss = actor_logprob[:, :slice_idx].unsqueeze(2) * advantage # B, H-1, 1
        else:
            raise NotImplementedError
        # - compute entropy regularized actor loss
        actor_entropy = actor_dist.entropy()
        entropy_scale = self.entropy_scale
        actor_loss = entropy_scale * actor_entropy[:, :slice_idx].unsqueeze(2) + actor_loss
        actor_loss = -(weights[:, :slice_idx] * actor_loss).mean()
        
        # Value loss
        self.value.train()
        self.value.requires_grad_(True)
        
        imagine_value_dist = self.value(imagine_feature[:,:slice_idx].detach())
        critic_logprob = -imagine_value_dist.log_prob(target.detach())
        critic_loss = weights[:,:slice_idx] * critic_logprob.unsqueeze(2)
        critic_loss = critic_loss.mean()
        imagine_value = imagine_value_dist.mean
        
        if global_step % self.log_every == 0:
            
            imagine_dist = Independent(OneHotCategorical(logits=imagine_state['logits']), 1)
            if self.actor_dist == 'onehot':
              action_samples = imagine_action.argmax(dim=-1).float().detach()
            else:
              action_samples = imagine_action.detach()
            logs = {
              'critic_loss': critic_loss.detach().item(),
              'actor_loss': actor_loss.detach().item(),
              'ACT_imagine_state': {k: v.detach() for k, v in imagine_state.items()},
              'ACT_imagine_entropy': imagine_dist.entropy().mean().detach().item(),
              'ACT_imagine_pcont': imagine_pcont.detach(),
              'ACT_imagine_value': imagine_value.squeeze(-1).detach(),
              'ACT_imagine_reward': imagine_reward.detach(),
              'ACT_imagine_idx': imagine_idx.float(),
              'ACT_target': target.squeeze(-1).detach(),
              'ACT_action_prob': actor_dist.mean.detach(),
              'ACT_action_samples': action_samples,
              'ACT_actor_target': target.mean().detach(),
              'ACT_actor_baseline': baseline.mean().detach(),
              'ACT_actor_entropy': actor_entropy.mean().item(),
              'ACT_actor_logprob': actor_logprob.mean().item(),
            }
        else:
          logs = {}
        
        return actor_loss, critic_loss, logs
    
    def update_slow_target(self):
        
        with torch.no_grad():
            if self.slow_update % self.slow_update_every_step == 0:
                self.slow_value.load_state_dict(self.value.state_dict())
            self.slow_update += 1
    
    def compute_target(self, imagine_feature, imagine_reward, imagine_pcont):
        
        self.slow_value.eval()
        self.slow_value.requires_grad_(False)
        
        # B, H, 1 TODO [REMINDER] : Check the dimension B*T, H, 1??
        imagine_slow_value = self.slow_value(imagine_feature).mean 
        
        # v_t = R_{t+1} + v_{t+1}
        target = self.lambda_return(imagine_reward[:, 1:], imagine_slow_value[:, :-1], imagine_pcont[:, 1:],
                                    imagine_slow_value[:, -1], self.lambda_)
        imagine_pcont = torch.cat([torch.ones_like(imagine_pcont[:, :1]), imagine_pcont[:, :-1]], dim=1)
        weights = torch.cumprod(imagine_pcont, 1).detach() # B, H, 1
        return target, weights
    
    def lambda_return(self, imagine_reward, imagine_value, discount, bootstrap, lambda_):
        """
        https://github.com/juliusfrost/dreamer-pytorch/blob/47bd509ab5cffa95ec613fd788d7ae1fe664ecd5/dreamer/algos/dreamer_algo.py
        """
        # Setting lambda=1 gives a discounted Monte Carlo return.
        # Setting lambda=0 gives a fixed 1-step return.
        next_values = torch.cat([imagine_value[:, 1:], bootstrap[:, None]], 1)
        target = imagine_reward + discount * next_values * (1 - lambda_)
        timesteps = list(range(imagine_reward.shape[1] - 1, -1, -1))

        outputs = []
        accumulated_reward = bootstrap
        for t in timesteps:

          inp = target[:, t]
          discount_factor = discount[:, t]

          accumulated_reward = inp + discount_factor * lambda_ * accumulated_reward
          outputs.append(accumulated_reward)

        returns = torch.flip(torch.stack(outputs, dim=1), [1])
        return returns
    
    def optimize_actor(self, actor_loss, actor_optimizer, writer, global_step):

      actor_loss.backward()
      grad_norm_actor = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)

      if (global_step % self.log_every == 0) and self.log_grad:
        for n, p in self.actor.named_parameters():
          if p.requires_grad:
            try:
              writer.add_scalar('grads/' + n, p.grad.norm(2), global_step)
            except:
              pdb.set_trace()

      actor_optimizer.step()

      return grad_norm_actor.item()
  
    def optimize_critic(self, value_loss, value_optimizer,writer , global_step):

        value_loss.backward()
        grad_norm_value = torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.grad_clip)

        if (global_step % self.log_every == 0) and self.log_grad:
          for n, p in self.value.named_parameters():
            if p.requires_grad:
              try:
                writer.add_scalar('grads/' + n, p.grad.norm(2), global_step)
              except:
                pdb.set_trace()
        value_optimizer.step()

        return grad_norm_value.item()
    
    
    # TODO [REMINDER] : Check this implementation
    def write_logs(self, logs, traj, global_step, writer, tag='train', min_idx=None):

        rec_img = logs['dec_img']
        gt_img = logs['gt_img']  # B, {1:T}, C, H, W

        writer.add_video('train/rec - gt',
                          torch.cat([gt_img[:4], rec_img[:4]], dim=-2).clamp(0., 1.).cpu(),
                          global_step=global_step)

        for k, v in logs.items():

          if 'loss' in k:
            writer.add_scalar(tag + '_loss/' + k, v, global_step=global_step)
          if 'grad_norm' in k:
            writer.add_scalar(tag + '_grad_norm/' + k, v, global_step=global_step)
          if 'hp' in k:
            writer.add_scalar(tag + '_hp/' + k, v, global_step=global_step)
          if 'ACT' in k:
            if isinstance(v, dict):
              for kk, vv in v.items():
                if isinstance(vv, torch.Tensor):
                  writer.add_histogram(tag + '_ACT/' + k + '-' + kk, vv, global_step=global_step)
                  writer.add_scalar(tag + '_mean_ACT/' + k + '-' + kk, vv.mean(), global_step=global_step)
                if isinstance(vv, float):
                  writer.add_scalar(tag + '_ACT/' + k + '-'  + kk, vv, global_step=global_step)
            else:
              if isinstance(v, torch.Tensor):
                writer.add_histogram(tag + '_ACT/' + k, v, global_step=global_step)
                writer.add_scalar(tag + '_mean_ACT/' + k, v.mean(), global_step=global_step)
              if isinstance(v, float):
                writer.add_scalar(tag + '_ACT/' + k, v, global_step=global_step)
          if 'imag_value' in k:
            writer.add_scalar(tag + '_values/' + k, v.mean(), global_step=global_step)
            writer.add_histogram(tag + '_ACT/' + k, v, global_step=global_step)
          if 'actor_target' in k:
            writer.add_scalar(tag + 'actor_target/' + k, v, global_step=global_step)         