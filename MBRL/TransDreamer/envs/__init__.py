from .box2d import Box2D
from .wrappers import OneHotAction, Collect, RewardObs
from .tools import save_episodes, summarize_episode

def make_env(cfg, writer, prefix, data_dir, store=True, render_mode='rgb_array'):
    
    env_suite, env_name = cfg.env.name.split('_', 1)
    time_limit = cfg.env.timelimit
    action_repeat = cfg.env.action_repeat
    seed = cfg.seed if cfg.reproducibility else 0
    continuous = cfg.env.continuous
    
    if env_suite == 'Box2D':
        env = Box2D(env_name, prefix, time_limit=time_limit, 
                    action_repeat=action_repeat, seed=seed, render_mode=render_mode, continuous=continuous)
    else:
        raise NotImplementedError(f"Environment {env_suite} not implemented")
    
    # Add OneHotAction wrapper
    env = OneHotAction(env)
    
    # Add Collect Wrapepr
    callbacks = []
    if store:
        callbacks.append(lambda ep: save_episodes(cfg, data_dir, [ep]))
    callbacks.append(lambda ep: summarize_episode(cfg, writer, prefix, data_dir, ep))
    env = Collect(env, callbacks=callbacks, precision=cfg.env.precision)
    
    # Add RewardObs Wrapper
    env = RewardObs(env)
    
    return env