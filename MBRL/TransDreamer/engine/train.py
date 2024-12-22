import os
import yaml
import torch

from pprint import pprint

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from solver import get_optimizer
from envs import make_env
from data import EnvIterDataset

from utils.utils import Checkpointer, print_colored
from envs.tools import count_steps

def train(model, cfg, device, verbose=1):
    
    print_colored("Prepare training...", "blue")
    # Verbose
    print_colored(f"- verbose level : {verbose}", "dark_white")
    if verbose >= 1:
        if verbose >= 2:
            pprint(cfg)
        pprint(model)
        
    # Model to device
    print_colored(f"- set model to {device}", "dark_white")
    model = model.to(device)
    
    # Optimizer
    print_colored(f"- set optimizer", "dark_white")
    optimizers = get_optimizer(cfg, model)
    
    # Relative paths
    print_colored(f"- create relative paths", "dark_white")
    root_dir = os.path.abspath(os.getcwd())
    checkpoint_dir = os.path.join(root_dir, cfg.checkpoint.checkpoint_dir)
    log_dir = os.path.join(root_dir, cfg.log_dir)
    data_dir = os.path.join(root_dir, cfg.data_dir)
    
    # Checkpointer
    print_colored(f"- create checkpointer", "dark_white")
    checkpointer_path = os.path.join(checkpoint_dir, cfg.exp_name, cfg.env.name, cfg.run_id)
    checkpointer = Checkpointer(checkpointer_path, max_num=cfg.checkpoint.max_num)
    with open(os.path.join(checkpointer_path, 'config.yaml'), 'w') as f:
        yaml.dump(cfg.to_dict(), f, default_flow_style=False)
        print_colored(f'- Saved configuration to {os.path.join(cfg.checkpoint.checkpoint_dir, "config.yaml")}', 
                      'dark_white')
    
    # Resume training
    if cfg.resume:
        checkpoint = checkpointer.load(cfg.resume_checkpoint)
        if checkpoint:
            model.load_state_dict(checkpoint['model'])
            for k, v in optimizers.items():
                if v is not None:
                    v.load_state_dict(checkpoint[k])
            env_step = checkpoint['env_step']
            global_step = checkpoint['global_step']
            print_colored(f"- resume training from env_step {env_step}, global_step {global_step}", "dark_blue")
        else:
            env_step = 0
            global_step = 0
            print_colored(f"- no checkpoint found, start training from scratch", "dark_yellow")
    else:
        env_step = 0
        global_step = 0
        print_colored(f"- start training from scratch", "dark_blue")
    
    # Tensorboard
    print_colored(f"- create tensorboard writer", "dark_white")
    log_dir = os.path.join(log_dir, cfg.exp_name, cfg.env.name, cfg.run_id)
    writer = SummaryWriter(log_dir=log_dir, flush_secs=30)
    
    # Environments
    print_colored(f"- create environments", "dark_white")
    data_dir = os.path.join(data_dir, cfg.exp_name, cfg.env.name, cfg.run_id, 'train_episodes')
    eval_data_dir = os.path.join(data_dir, cfg.exp_name, cfg.env.name, cfg.run_id, 'test_episodes')
    train_env = make_env(cfg, writer, 'train', data_dir, store=True, render_mode=cfg.env.render_mode_train)
    test_env = make_env(cfg, writer, 'test', eval_data_dir, store=True, render_mode=cfg.env.render_mode_test)
    
    # Prefill
    train_env.reset()
    steps = count_steps(cfg, data_dir)
    if steps < cfg.arch.prefill_steps:
        print_colored(f"- prefill the replay buffer", "dark_blue")
        while steps < cfg.arch.prefill_steps:
            action = train_env.sample_random_action()
            _, _, terminated, truncated, _ = train_env.step(action)
            if terminated or truncated:
                steps += train_env.step_count
                train_env.reset()
        print_colored(f"- prefill done", "dark_blue")
    else:
        print_colored(f"- prefill already done", "dark_yellow")
    
    steps = count_steps(cfg, data_dir)
    print_colored(f"- collected {steps} steps", "dark_white")
    
    # Dataset and Dataloader
    train_dataset = EnvIterDataset(data_dir, cfg.train.train_steps, cfg.train.episode_length, cfg.seed)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers)
    train_iter = iter(train_dataloader)
    global_step = max(global_step, steps)
    
    # Initialize for training
    obs, _ = train_env.reset()
    state = None # posterior state
    action_list = torch.zeros(1, 1, cfg.env.action_size).float() # B, T, d_action
    action_list[0, 0, 0] = 1.
    obs_type = cfg.arch.world_model.obs_type
    total_steps = int(float(cfg.train.total_steps))
    train_every = cfg.train.train_every
    eval_every = int(float(cfg.train.eval_every))
    log_every = int(float(cfg.train.log_every))
    checkpoint_every = int(float(cfg.train.checkpoint_every))
    episode_length = cfg.train.episode_length
    
    print_colored("Check training configuration...", "blue")
    print_colored(f"- gloabl_step: {global_step}", "dark_white")
    print_colored(f"- env_step: {env_step}", "dark_white")
    print_colored(f"- total_steps: {total_steps}", "dark_white")
    print_colored(f"- train_every: {train_every}, type: {type(train_every)}", "dark_white")
    print_colored(f"- log_every: {log_every}, type: {type(log_every)}", "dark_white")
    print_colored(f"- eval_every: {eval_every}, type: {type(eval_every)}", "dark_white")
    print_colored(f"- checkpoint_every: {checkpoint_every}, type: {type(checkpoint_every)}", "dark_white")
    
    print_colored("Start training loop...", "blue")
    global_step_count = train_env.step_count
    while global_step < total_steps:
        
        # Exploration
        with torch.no_grad():
            
            print_colored(f"- Exploration : {global_step}/{total_steps}", "dark_white")
            model.eval()
            next_obs, _, terminated, truncated, _ = train_env.step(action_list[0, -1].detach().cpu().numpy())
            prev_image = torch.tensor(obs[obs_type])
            next_image = torch.tensor(next_obs[obs_type])
            action_list, state = model.policy(prev_image.to(device), next_image.to(device), action_list.to(device),
                                              state=state, episode_length=episode_length)
            obs = next_obs
            if truncated or terminated:
                obs, _ = train_env.reset() # TODO [IMPORTANT] : obs should be updated
                state = None
                action_list = torch.zeros(1, 1, cfg.env.action_size).float() # B, T, A = d_action
                action_list[0, 0, 0] = 1.
        
        # Training
        if global_step % train_every == 0:
            
            print_colored(f"🔥 Training", "dark_green")
            model.train()
            logs = {}
            
            # Load data
            traj = next(train_iter)
            for k, v in traj.items():
                traj[k] = v.to(device).float()
        
            # World model training
            model_optimizer = optimizers['model_optimizer']
            model_optimizer.zero_grad()
            model_loss, model_logs, _, post_state = model.world_model_loss(traj, global_step)
            grad_norm_model = model.world_model.optimize_world_model(model_loss, model_optimizer, writer, global_step)
            
            # Actor-Critic training
            actor_optimizer = optimizers['actor_optimizer']
            critic_optimizer = optimizers['critic_optimizer']
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()
            actor_loss, critic_loss, actor_critic_logs = model.actor_and_critic_loss(traj, post_state, global_step)
            grad_norm_actor = model.optimize_actor(actor_loss, actor_optimizer, writer, global_step)
            grad_norm_critic = model.optimize_critic(critic_loss, critic_optimizer, writer, global_step)
            
            # Loggings
            if global_step % log_every == 0:
                 
                print_colored(f"📥 Logging", "dark_green")
                logs.update(model_logs)
                logs.update(actor_critic_logs)
                # model.write_logs(logs, traj, global_step, writer) # TODO [REMINDER] : Later implementation
                
                grad_norm = dict(
                    grad_norm_model = grad_norm_model,
                    grad_norm_actor = grad_norm_actor,
                    grad_norm_critic = grad_norm_critic,
                )
                for k, v in grad_norm.items():
                    writer.add_scalar(f"train_grad_norm/" + k, v, global_step = global_step)
        
        # Evaluation -> TODO [REMINDER] : Revise as TEST
        # if global_step % eval_every == 0:
        #     print_colored(f"📈 Evaluation", "dark_green")
        #     # TODO
        #     simulate_test(cfg, model, test_env, global_step, device) 
            
        # Checkpoint
        if global_step % checkpoint_every == 0:
            print_colored(f"💾 Checkpoint", "dark_green")
            checkpointer.save(model, optimizers, global_step, env_step)

        # Update global step        
        global_step += train_env.step_count - global_step_count
        global_step_count = train_env.step_count


def simulate_test(cfg, model, test_env, global_step, device):

    model.eval()

    obs, _ = test_env.reset()
    action_list = torch.zeros(1, 1, cfg.env.action_size).float()
    action_list[:, 0, 0] = 1. # B, T, C
    state = None
    truncated = False
    terminated = False
    obs_type = cfg.arch.world_model.obs_type

    with torch.no_grad():
        while not (truncated or terminated):
          next_obs, reward, terminated, trucated, _ = test_env.step(action_list[0, -1].detach().cpu().numpy())
          prev_image = torch.tensor(obs[obs_type])
          next_image = torch.tensor(next_obs[obs_type])
          action_list, state = model.policy(prev_image.to(device), 
                                            next_image.to(device), 
                                            action_list.to(device), 
                                            state, 
                                            training=False, 
                                            episode_length=cfg.train.episode_length)
          obs = next_obs