import gym
from gym.spaces import Box

import numpy as np


class Probe1(gym.Env):
    """
    Constant state, reward always equals 1, 1 steo=p
    """
    def __init__(self):
        self.observation_space = Box(1, 1, (1,))
        self.action_space = Box(-1, 1, (1,))

    def step(self, action):
        return np.array((1,)), 1, 1, {}

    def reset(self):
        return np.array((1,))

    def render(self, mode='human'):
        pass


class Probe2(gym.Env):
    """
    Constant state, reward equals action
    """
    def __init__(self):
        self.observation_space = Box(-1, 1, (1,))
        self.action_space = Box(-1, 1, (1,))

    def step(self, action: np.ndarray):
        return np.array((0,)), action[0], 1, {}

    def reset(self):
        return np.array((0,))

    def render(self, mode='human'):
        pass


class Probe3(gym.Env):
    """
    State and reward dependent on time
    """
    def __init__(self):
        self.observation_space = Box(0, 5, (1,))
        self.action_space = Box(-1, 1, (1,))
        self.t = 0

    def step(self, action: np.ndarray):
        self.t += 1
        return np.array((self.t,)), self.t, self.t >= 5, {}

    def reset(self):
        self.t = 0
        return np.array((0,))

    def render(self, mode='human'):
        pass


class Probe4(gym.Env):
    """
    Reward equals the state
    """
    def __init__(self):
        self.observation_space = Box(-1, 1, (1,))
        self.action_space = Box(-1, 1, (1,))
        self.last_obs = None

    def step(self, action: np.ndarray):
        reward = self.last_obs[0]
        self.last_obs = self.observation_space.sample()
        return self.last_obs, reward, 1, {}

    def reset(self):
        self.last_obs = self.observation_space.sample()
        return self.last_obs

    def render(self, mode='human'):
        pass


probes = {
    'p1': Probe1,
    'p2': Probe2,
    'p3': Probe3,
    'p4': Probe4,
}
