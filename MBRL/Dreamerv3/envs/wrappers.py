import datetime
import gymnasium as gym
import numpy as np
import uuid
import cv2
import os
import pathlib

# class RewardObs(gym.Wrapper):
#     def __init__(self, env):
#         super().__init__(env)
#         spaces = self.env.observation_space.spaces
#         if "obs_reward" not in spaces:
#             spaces["obs_reward"] = gym.spaces.Box(
#                 -np.inf, np.inf, shape=(1,), dtype=np.float32
#             )
#         self.observation_space = gym.spaces.Dict(spaces)

#     def step(self, action):
#         obs, reward, terminated, truncated, info = self.env.step(action)
#         if "obs_reward" not in obs:
#             obs["obs_reward"] = np.array([reward], dtype=np.float32)
#         done = terminated or truncated
#         return obs, reward, done, info

#     def reset(self, **kwargs):
#         obs = self.env.reset(**kwargs)
#         if "obs_reward" not in obs:
#             obs["obs_reward"] = np.array([0.0], dtype=np.float32)
#         return obs


class SelectAction(gym.Wrapper):
    def __init__(self, env, key):
        super().__init__(env)
        self._key = key

    def step(self, action):
        return self.env.step(action[self._key])


class TerminateOutsideTrackWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        
        obs, reward, done, info = self.env.step(action)
        
        if self._is_all_green(obs):
            done = True
            reward -= 100
        
        return obs, reward, done, info
    
    def _is_all_green(self, obs):
        green_rgb = np.array([100, 220, 100])
        tolerance = 10 
        
        diff = np.abs(obs["image"] - green_rgb).mean(axis=-1)
        green_pixels = np.sum(diff < tolerance)
        
        return green_pixels > obs["image"].shape[0] * obs["image"].shape[1] * 0.865
    
class UUID(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        self.id = f"{timestamp}-{str(uuid.uuid4().hex)}"

    def reset(self, **kwargs):
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        self.id = f"{timestamp}-{str(uuid.uuid4().hex)}"
        return self.env.reset()
    
class VideoRecorder:
    def __init__(self, video_path, frame_size, fps=30):

        self.video_path = video_path
        self.frame_size = frame_size
        self.fps = fps
        self.writer = None

        directory = pathlib.Path(self.video_path).parent
        directory.mkdir(parents=True, exist_ok=True)

    def start_recording(self):

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # mp4 포맷
        self.writer = cv2.VideoWriter(self.video_path, fourcc, self.fps, self.frame_size)

    def write_frame(self, frame):

        if self.writer is not None:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # OpenCV는 BGR 사용
            self.writer.write(frame_bgr)

    def stop_recording(self):
 
        if self.writer is not None:
            self.writer.release()
            self.writer = None

class VideoCaptureWrapper:
    def __init__(self, env, video_recorder):
        self.env = env
        self.video_recorder = video_recorder

    def reset(self):
        obs = self.env.reset()

        frame = obs["image"]

        self.video_recorder.write_frame(frame)

        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        frame = obs["image"]

        self.video_recorder.write_frame(frame)

        return obs, reward, done, info

    def close(self):
        self.env.close()
        self.video_recorder.stop_recording()

    def __getattr__(self, name):
        return getattr(self.env, name)

