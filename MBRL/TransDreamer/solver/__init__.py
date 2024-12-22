from torch import optim

def get_optimizer(cfg, model):
    
    optimizer = cfg.optimize.optimizer
    kwargs = {'weight_decay': float(cfg.optimize.weight_decay), 
              'eps': float(cfg.optimize.eps)}
    model_lr = float(cfg.optimize.model_lr)
    actor_lr = float(cfg.optimize.actor_lr)
    critic_lr = float(cfg.optimize.critic_lr)
    
    if optimizer == "adam":
        optimizer = optim.Adam
    elif optimizer == "adamW":
        optimizer = optim.AdamW

    # TODO [REMINDER] I removed warm-up process in transformer
    model_optimizer = optimizer(model.world_model.parameters(), lr=model_lr, **kwargs)
    actor_optimizer = optimizer(model.actor.parameters(), lr=actor_lr, **kwargs)
    critic_optimizer = optimizer(model.value.parameters(), lr=critic_lr, **kwargs)
    
    return {
        'model_optimizer': model_optimizer,
        'actor_optimizer': actor_optimizer,
        'critic_optimizer': critic_optimizer
    }
    
    
    