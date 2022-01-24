import rsa.utils.plot_utils as pu

import gym
#from gym.wrappers import LazyFrames
import os
import numpy as np
import moviepy.editor as mpy


class SimpleVideoSaver(gym.Wrapper):
    def __init__(self, env: gym.Env, video_dir, 
                 from_render=False, 
                 speedup=1,
                 camera=None):
        super().__init__(env)
        self.env = env
        self.dir = video_dir
        self.from_render = from_render
        self.speedup = speedup
        self.camera = camera

        os.mkdir(self.dir)
        self.video_buffer = []
        self.count = 0

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        if self.from_render:
            next_obs_in = np.flip(self.env.sim.render(500, 500, camera_name=self.camera), axis=0)
            # next_obs_in = self.env.render(mode='rgb_array')
        else:
            if type(next_obs) == LazyFrames:
                next_obs_in = next_obs[0]
            else:
                next_obs_in = next_obs
            next_obs_in = next_obs_in.transpose((1, 2, 0))
        self.video_buffer.append(next_obs_in)
        return next_obs, reward, done, info

    def reset(self, **kwargs):
        if len(self.video_buffer) > 0:
            self._make_movie()
        self.video_buffer = []
        self.count += 1

        obs = self.env.reset(**kwargs)
        if self.from_render:
            obs_in = np.flip(self.env.sim.render(500, 500, camera_name=self.camera), axis=0)
            # obs_in = self.env.render(mode='rgb_array')
        else:
            #if type(obs) == LazyFrames:
             #   obs_in = obs[0]
            #else:
            obs_in = obs
            obs_in = obs_in.transpose((1, 2, 0))
        self.video_buffer.append(obs_in)

        return obs

    def dump_movie(self):
        self._make_movie()
        self.video_buffer = []

    def _make_movie(self):
        file = os.path.join(self.dir, '%d.mp4' % self.count)
        pu.make_movie(self.video_buffer, file, speedup=self.speedup)
