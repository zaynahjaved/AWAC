import rsa.utils as utils

import gym


class InfoWrapper(gym.Wrapper):
    def __init__(self, env):
        super(InfoWrapper, self).__init__(env)
        if type(env) == TimeLimitWrapper:
            print(utils.colorize('WARNING: Wrapping InfoWrapper around TimeLimitWrapper '
                                 'will mess with masks', 'red', True))

    """Adds extra info that we assume will be in there. Assumes it is never in the goal and
    mask alwasy is the inverse of goal"""
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        info['goal'] = 0
        info['mask'] = not done
        return observation, reward, done, info


class TimeLimitWrapper(gym.Wrapper):
    def __init__(self, env, time_limit):
        super(TimeLimitWrapper, self).__init__(env)
        self.time_limit = time_limit
        self.t = 0

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.t += 1
        if self.t >= self.time_limit:
            done = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self.t = 0
        return observation


class HalfCheetahWrapper(gym.Wrapper):
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        info['goal'] = 0
        info['mask'] = 1
        return observation, reward, done, info


class D4RLWrapper(gym.Wrapper):
    def __init__(self, env, threshold=0):
        super(D4RLWrapper, self).__init__(env)
        self.threshold = threshold

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward = (reward > self.threshold) - 1
        return observation, reward, done, info
