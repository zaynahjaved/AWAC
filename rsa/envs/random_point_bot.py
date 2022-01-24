"""
A robot that can exert force in cardinal directions. The robot's goal is to
reach the origin and it experiences zero-mean Gaussian Noise. State
representation is (x, y). Action representation is (dx, dy).
"""
import gym

from rsa.envs.simple_point_bot import SimplePointBot

import numpy as np
from gym import Env
from gym import utils
from gym import Wrapper
from gym.spaces import Box

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import os

"""
Constants associated with the PointBot env.
"""

WINDOW_WIDTH = 180
WINDOW_HEIGHT = 150
WINDOW_SIZE = (180, 150)

MAX_FORCE = 3


class RandomPointBot(Env, utils.EzPickle):

    def __init__(self, from_pixels=False,
                 horizon=100,
                 goal_thresh=3,
                 noise_scale=0.125):
        utils.EzPickle.__init__(self)
        self.pos = self.goal = self.walls = self.start = None
        self.horizon = horizon
        self.goal_thresh = goal_thresh
        self.noise_scale = noise_scale
        self.window_width = WINDOW_WIDTH
        self.window_height = WINDOW_HEIGHT
        self.max_force = MAX_FORCE
        self.action_space = Box(-np.ones(2), np.ones(2))
        self.observation_space = Box(-np.ones(4) * np.float('inf'), np.ones(4) * np.float('inf'))
        self._episode_steps = 0
        # self.obstacle = self._complex_obstacle(OBSTACLE_COORDS)
        self._from_pixels = from_pixels
        self._image_cache = {}

    def step(self, act):
        act = np.clip(act, -1, 1) * MAX_FORCE

        next_pos = self.pos + act + self.noise_scale * np.random.randn(len(self.pos))
        next_pos = np.clip(next_pos, (0, 0), (WINDOW_WIDTH, WINDOW_HEIGHT))

        in_goal = int(np.linalg.norm(np.subtract(self.goal, next_pos)) < self.goal_thresh)
        reward = in_goal - 1

        self.pos = next_pos
        self._episode_steps += 1
        done = self._episode_steps >= self.horizon or in_goal
        mask = 1 - in_goal

        obs = self._get_obs()
        return obs, reward, done, {
            'goal': in_goal,
            'mask': mask
        }

    def reset(self, random_start=False):
        self.pos = np.random.random(2) * (WINDOW_WIDTH, WINDOW_HEIGHT)
        # self.pos = (20, 75)
        self.start = self.pos
        self.goal = np.random.random(2) * (WINDOW_WIDTH, WINDOW_HEIGHT)
        # self.goal = (130, 75)
        self._episode_steps = 0
        obs = self._get_obs()
        return obs

    def render(self, mode='human'):
        pass
        # return self._draw_state(self.pos)

    def draw_board(self, ax):
        plt.xlim(0, WINDOW_WIDTH)
        plt.ylim(0, WINDOW_HEIGHT)

        circle = plt.Circle(self.start, radius=3, color='k')
        ax.add_patch(circle)
        circle = plt.Circle(self.goal, radius=3, color='k')
        ax.add_patch(circle)
        ax.annotate("goal", xy=(self.goal[0], self.goal[1] - 8), fontsize=10, ha="center")

    def _get_obs(self):
        return np.array((*self.pos, *self.goal)) / np.array((*WINDOW_SIZE, *WINDOW_SIZE))


def expert_pol(obs):
    pos = obs[:2]
    goal = obs[2:]
    diff = goal - pos
    return diff / np.max(np.abs(diff))
