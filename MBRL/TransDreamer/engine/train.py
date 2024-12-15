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
    test_env = make_env(cfg, writer, 'test', eval_data_dir, store=True, render_mode=cfg.env.render_mode_eval)
    
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
    state = None
    action_list = torch.zeros(1, 1, cfg.env.action_size).float() # B, T, d_action
    action_list[0, 0, 0] = 1.
    obs_type = cfg.arch.world_model.obs_type
    total_steps = int(float(cfg.train.total_steps))
    train_every = int(float(cfg.train.train_every))
    log_every = int(float(cfg.train.log_every))
    eval_every = int(float(cfg.train.eval_every))
    checkpoint_every = int(float(cfg.train.checkpoint_every))
    
    print_colored("Check training configuration...", "blue")
    print_colored(f"- gloabl_step: {global_step}", "dark_white")
    print_colored(f"- env_step: {env_step}", "dark_white")
    print_colored(f"- total_steps: {total_steps}", "dark_white")
    print_colored(f"- train_every: {train_every}", "dark_white")
    print_colored(f"- log_every: {log_every}", "dark_white")
    print_colored(f"- eval_every: {eval_every}", "dark_white")
    print_colored(f"- checkpoint_every: {checkpoint_every}", "dark_white")
    
    # print_colored("Start training loop...", "blue")
    # while global_step < total_steps:
        
    #     # Evaluation
    #     if global_step % 
    #     print_colored("- Evaluation", "green")
    #     with torch.no_grad():
    #         model.eval()
            
            
    #     # Training
    #     print_colored("- Training", "green")
    #     pass