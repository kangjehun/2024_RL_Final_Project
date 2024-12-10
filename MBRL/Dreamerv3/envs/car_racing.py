import gymnasium as gym
import numpy as np
import cv2

class CarRacing:
    def __init__(self, size=(64, 64), gray=False, seed=None, render=False):
        self._size = (64, 64)
        self._repeat = 4
        self._gray = gray
        self._random = np.random.RandomState(seed)
        self._env = gym.make("CarRacing-v2", render_mode="rgb_array")
        self._done = True
        self._step = 0
        self._is_first = True
        self._render = render

    @property
    def observation_space(self):
        img_shape = self._size + ((1,) if self._gray else (3,))
        return gym.spaces.Dict({"image": gym.spaces.Box(low=0, high=255, shape=img_shape, dtype=np.uint8)})

    @property
    def action_space(self):
        return self._env.action_space

    def reset(self):
        obs, info = self._env.reset(seed=self._random.randint(0, 10000))
        self._done = False
        self._step = 0
        self._is_first = True
        processed_obs = self._process_obs(obs)
        return {"image": processed_obs, "is_first": self._is_first, "is_terminal": False}

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        
        for _ in range(self._repeat):
            obs, reward, terminated, truncated, info = self._env.step(action)
            total_reward += reward
            if self._render:
                frame = self._env.render()
                pass
            if terminated or truncated:
                self._done = True
                break

        processed_obs = self._process_obs(obs)
        step_obs = {
            "image": processed_obs,
            "is_first": self._is_first,
            "is_terminal": self._done,
        }
        self._is_first = False
        done = terminated or truncated
        return step_obs, total_reward, done, info

    def _process_obs(self, obs):
        if self._gray:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
            obs = cv2.resize(obs, self._size, interpolation=cv2.INTER_AREA)
            obs = obs[:, :, None] 
        else:
            obs = cv2.resize(obs, self._size, interpolation=cv2.INTER_AREA)
        return obs

    def close(self):
        self._env.close()
