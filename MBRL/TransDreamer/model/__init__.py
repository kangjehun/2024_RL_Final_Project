from .transdreamer import TransDreamer

def get_model(cfg, device, seed=0):
    
    if cfg.model == "transdreamer":
        model = TransDreamer(cfg)
        
    return model