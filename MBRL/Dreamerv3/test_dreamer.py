import argparse
import pathlib
import sys
import torch
import numpy as np
import ruamel.yaml as yaml
from dreamer import Dreamer
import tools
import functools
from parallel import Damy
import cv2

sys.path.append(str(pathlib.Path(__file__).parent))

import envs.wrappers as wrappers
to_np = lambda x: x.detach().cpu().numpy()

class VideoRecorder:
    def __init__(self, video_path, frame_size, fps=30):
        self.video_path = video_path
        self.frame_size = frame_size
        self.fps = fps
        self.writer = None

    def start_recording(self):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v") 
        self.writer = cv2.VideoWriter(self.video_path, fourcc, self.fps, self.frame_size)

    def write_frame(self, frame):
        if self.writer is not None:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  
            self.writer.write(frame_bgr)

    def stop_recording(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            

def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))

def make_env(record_video=True, video_path="./videos/episode.mp4"):
    from envs.car_racing import CarRacing

    env = CarRacing()

    env = wrappers.TerminateOutsideTrackWrapper(env)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)

    return env

    

def make_dataset(episodes, config):
    generator = tools.sample_episodes(episodes, config.batch_length)
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset

def test_agent(config, run_id):
    tools.set_seed_everywhere(config.seed)
    # logdir = pathlib.Path(f"./logdir/test/run_{run_id}").expanduser() 
    logdir = pathlib.Path("./logdir/test").expanduser() 
    config.evaldir = logdir / "eval_eps"
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat

    logdir.mkdir(parents=True, exist_ok=True)

    step = count_steps(config.evaldir)
    logger = tools.Logger(logdir, config.action_repeat * step)

    eval_env = [make_env(record_video=True, video_path="./videos/episode.mp4")]
    eval_env = [Damy(env) for env in eval_env]
    eval_eps = tools.load_episodes(config.evaldir, limit=1) 

    acts = eval_env[0].action_space
    config.num_actions = acts.shape[0] if hasattr(acts, "shape") else acts.n
    print(f"Number of actions: {config.num_actions}")

    agent = Dreamer(
        eval_env[0].observation_space,
        eval_env[0].action_space,
        config,
        logger, 
        dataset=None  
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)

    model_path = pathlib.Path("./logdir/car_racing")
    model = torch.load(model_path / "10million.pt", weights_only=False)
    agent.load_state_dict(model["agent_state_dict"])
    tools.recursively_load_optim_state_dict(agent, model["optims_state_dict"])
    agent._should_pretrain._once = False

    eval_policy = functools.partial(agent, training=False)
    eval_dataset = make_dataset(eval_eps, config)

    print("Start evaluation.")
    logger.write()
    tools.simulate(
        eval_policy,
        eval_env,
        eval_eps,
        config.evaldir,
        logger,
        is_eval=True,
        episodes=1
    )
    video_pred = agent._wm.video_pred(next(eval_dataset))
    logger.video("eval_openl", to_np(video_pred))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    args, remaining = parser.parse_known_args()
    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
    )

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))

    for run_id in range(1): 
        test_agent(parser.parse_args(remaining), run_id)
