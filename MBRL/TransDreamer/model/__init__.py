from .transdreamer import TransDreamer

def get_model(cfg):
    
    if cfg.model == "transdreamer":
        model = TransDreamer(cfg)
        
    return model