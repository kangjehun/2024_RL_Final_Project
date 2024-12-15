import pathlib

import numpy as np

from torch.utils.data import IterableDataset, get_worker_info

from utils.utils import print_colored


class EnvIterDataset(IterableDataset):
    
    def __init__(self, data_dir, train_steps, episode_length, seed=0):
        
        self.data_dir = data_dir
        self.train_steps = train_steps
        self.episode_length = episode_length
        self.seed = seed
        
    # TODO : Revise the code to complete self.train_steps while considering skipping short episodes
    def load_episodes(self, balance=False):
                
        directory = pathlib.Path(self.data_dir).expanduser()
        worker_info = get_worker_info()
        if worker_info is None:
            raise ValueError("No worker info")
        random = np.random.RandomState((self.seed + worker_info.seed) % (1 << 32))
        cache = {}
        
        while True:
        
            # Load filenames from the directory to the cache
            for filename in directory.glob("*.npz"):
                if filename not in cache:
                    cache[filename] = filename
                
            # Randomly select episodes
            keys = list(cache.keys())
            indices = random.choice(len(keys), size=self.train_steps)
            
            # Load the episodes
            for index in indices:
                filename = cache[keys[index]]
                try:
                    with open(filename, 'rb') as f:
                        episode = np.load(f)
                        episode = {k: episode[k] for k in episode.keys()}
                except Exception as e:
                    print_colored(f"Could not load episode: {e}", "red")
                    continue
                
                if self.episode_length:
                    total = len(next(iter(episode.values())))
                    available = total - self.episode_length
                    if available < 1 :
                        print_colored(f"Skipped short episode ({filename}) of available {available}", "yellow")
                        continue
                    if balance:
                        raise NotImplementedError("Balancing not implemented yet") # TODO [REMINDER] Not necessary
                    else:
                        index = int(random.randint(0, available))
                    episode = {k: v[index : index + self.episode_length] for k, v in episode.items()}
                
                yield episode
    
    def __iter__(self):
        return self.load_episodes()