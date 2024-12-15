import pathlib
import datetime
import uuid
import io
import glob
import os
import json
import torch

import numpy as np
from utils.utils import print_colored

def save_episodes(cfg, directory, episodes):
    directory = pathlib.Path(directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    for episode in episodes:
        identifier = str(uuid.uuid4().hex)
        length = len(episode['reward']) * cfg.env.action_repeat
        if 'obs_reward' in episode:
            raise NotImplementedError
        else:
            abs_reward = sum(episode['reward'])
        filename = directory / f'{timestamp}_{identifier}_{length}_{abs_reward}.npz'
        with io.BytesIO() as f1:
            np.savez_compressed(f1, **episode)
            f1.seek(0)
            with filename.open('wb') as f2:
                f2.write(f1.read())

def summarize_episode(cfg, writer, prefix, data_dir, episode):
    episodes, _ = count_episodes(cfg, data_dir)
    length = (len(episode['reward']) - 1) * cfg.env.action_repeat
    ret = episode['reward'].sum()
    print_colored(f'✔ {prefix.title()} episode of length {length} with return {ret:.1f}', 'dark_white')
    metrics = [
        (f'{prefix}/return', float(episode['reward'].sum())),
        (f'{prefix}/length', len(episode['reward']) - 1),
        (f'{prefix}/episodes', episodes)
        ]
    step = count_steps(cfg, data_dir)
    env_step = step # TODO [IMPORTANT] Store exact env_step in the episode!!
    with (pathlib.Path(cfg.log_dir) / 'metrics.jsonl').open('a') as f:
        f.write(json.dumps(dict([('step', env_step)] + metrics)) + '\n')
    [writer.add_scalar('sim/' + k, v, env_step) for k, v in metrics]
    if 'bev' in episode:
        video_summary(writer, f'sim/{prefix}/video', episode['bev'][None, :1000], env_step)
    else:
        print_colored('\U000026A0 No BEV in episode', 'dark_yellow')
    
    if 'episode_done' in episode:
        raise NotImplementedError # TODO [REMINDER] it looks not necessary
        episode_done = episode['episode_done']
        num_episodes = sum(episode_done)
        writer.add_scalar(f'sim/{prefix}/num_episodes', num_episodes, env_step)
        # compute sub-episode len
        episode_done = np.insert(episode_done, 0, 0)
        episode_len_ = np.where(episode_done)[0]
        if len(episode_len_) > 1:
            episode_len_ = np.insert(episode_len_, 0, 0)
            episode_len_ = episode_len_[1:] - episode_len_[:-1]
            writer.add_histogram(f'sim/{prefix}/sub_episode_len', episode_len_, env_step)
            writer.add_scalar(f'sim/{prefix}/sub_episode_len_min', episode_len_[1:].min(), env_step)
            writer.add_scalar(f'sim/{prefix}/sub_episode_len_max', episode_len_[1:].max(), env_step)
            writer.add_scalar(f'sim/{prefix}/sub_episode_len_mean', episode_len_[1:].mean(), env_step)
            writer.add_scalar(f'sim/{prefix}/sub_episode_len_std', episode_len_[1:].std(), env_step)

    writer.flush() 
    
def count_episodes(cfg, directory):
    """ Count the number of episodes and steps in a directory of episodes """
    # filenames = directory.glob('*.npz')
    filenames = glob.glob(os.path.join(directory, '*.npz'))
    filenames = [pathlib.Path(f) if isinstance(f, str) else f for f in filenames ]
    lengths = []
    for f in filenames:
        try:
            lengths.append(int(f.stem.rsplit('_', 2)[1]) - cfg.env.action_repeat) # subtract 1 for reset (initial state)
        except (IndexError, ValueError):
            print_colored(f'Error parsing episode length from {f}', 'red')
    episodes, steps = len(lengths), sum(lengths)
    return episodes, steps

def count_steps(cfg, directory):
    return count_episodes(cfg, directory)[1]

def video_summary(writer, name, video, step=None):
    name = name if isinstance(name, str) else name.decode('utf-8')
    if np.issubdtype(video.dtype, np.floating):
        video = np.clip(255 * video, 0, 255).astype(np.uint8)
    writer.add_video(
        name,
        torch.tensor(video, dtype=torch.uint8),
        step
    )