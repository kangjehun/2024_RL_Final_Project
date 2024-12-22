import torch

class SafeTruncatedNormal(torch.distributions.normal.Normal):
    
    def __init__(self, loc, scale, low, high, clip=1e-6, mult=1):
        
        super().__init__(loc, scale)
        
        self._low = low
        self._high = high
        self._clip = clip
        self._mult = mult

    def sample(self, sample_shape):
        
        event = super().sample(sample_shape)
        if self._clip:
            clipped = torch.clamp(event, self._low + self.clip, self._high - self.clip)
            event = event - event.detach() + clipped
        if self._mult:
            event *= self._mult
        return event

class ContinuousDist:
    
    def __init__(self, dist=None):
        
        super().__init__()
        self.dist = dist
        self.mean = dist.mean
        
    def __getattr__(self, name):
        return getattr(self._dist, name)
    
    def entropy(self):
        return self._dist.entropy()
    
    def mode(self):
        return self._dist.mean()
    
    def sample(self, sample_shape=()):
        return self._dist.rsample(sample_shape) # TODO [REMINDER] Suspicious Implementation
    
    def log_prob(self, x):
        return self._dist.log_prob(x)
    
        