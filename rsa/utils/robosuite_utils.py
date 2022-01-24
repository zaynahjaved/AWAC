from rsa.envs.normalized_box_env import NormalizedBoxEnv
from rsa.envs.simple_video_saver import SimpleVideoSaver

import gym
import numpy as np
import moviepy.editor as mpy
import os
from skimage.transform import resize

import robosuite as suite
from robosuite.wrappers import GymWrapper
from robosuite import load_controller_config
from robosuite.utils.transform_utils import pose2mat


# class TerminateOnCompleteWrapper(gym.Wrapper):
#     """
#     Makes it terminate if rew > 0
#     """
#     def __init__(self, env):
#         super().__init__(env)
#         self.env = env
#
#     def step(self, act):
#         next_obs, rew, done, info = self.env.step(act)
#         done = done or rew > 0
#         return next_obs, rew, done, info


class RSWrapper(gym.Wrapper):
    """
    Infers goal info, adds mask info (always 1), and shifts reward down by -1
    """
    def step(self, action):
        next_obs, rew, done, info = super(RSWrapper, self).step(action)
        info['goal'] = rew
        info['mask'] = 1
        rew -= 1
        return next_obs, rew, done, info


class RSVisualizationGymWrapper(GymWrapper):
    """
    Modified version of
    https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/wrappers/gym_wrapper.py
    to help with visualization
    """

    def __init__(self, env, video_dir, keys=None):
        # Run super method
        if keys is None:
            keys = [
                'object-state', 'robot0_proprio-state'
            ]
        super().__init__(env=env, keys=keys)

        self.dir = video_dir
        if not os.path.exists(self.dir):
            os.mkdir(self.dir)
        self.video_buffer = []
        self.count = 0

    def reset(self):
        """
        Extends env reset method to return flattened observation instead of normal OrderedDict.
        Returns:
            np.array: Flattened environment observation space after reset occurs
        """
        if len(self.video_buffer) > 0:
            self._make_movie()
        self.video_buffer = []
        self.count += 1

        ob_dict = self.env.reset()
        self._append_to_dir(ob_dict)
        return self._flatten_obs(ob_dict)

    def step(self, action):
        """
        Extends vanilla step() function call to return flattened observation instead of normal OrderedDict.

        Args:
            action (np.array): Action to take in environment
        Returns:
            4-tuple:

                - (np.array) flattened observations from the environment
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) misc information
        """
        ob_dict, reward, done, info = self.env.step(action)
        self._append_to_dir(ob_dict)
        return self._flatten_obs(ob_dict), reward, done, info

    def _append_to_dir(self, ob_dict):
        im = ob_dict['agentview_image']
        im = np.flip(im, axis=0)
        # im = (resize(im, (64, 64)) * 255).astype(np.uint8)
        self.video_buffer.append(im)

    def _make_movie(self):
        file = os.path.join(self.dir, '%d.mp4' % self.count)
        clip = mpy.ImageSequenceClip(self.video_buffer, fps=10)
        clip.write_videofile(file, logger=None)


class RSGymWrapper(GymWrapper):
    """
    Modified version of
    https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/wrappers/gym_wrapper.py
    to do the following:
        > add state data to the info array returned alongside the image-based observation
    """

    def __init__(self, env, keys=None):
        # Run super method
        if keys is None:
            keys = [
                'object-state', 'robot0_proprio-state'
            ]
        super().__init__(env=env, keys=keys)
        self.state = None

    def reset(self, **kwargs):
        """
        Extends env reset method to return flattened observation instead of normal OrderedDict.
        Returns:
            np.array: Flattened environment observation space after reset occurs
        """
        ob_dict = self.env.reset()
        self.state = self._flatten_obs(ob_dict)
        return self._process_image(ob_dict)

    def step(self, action):
        """
        Extends vanilla step() function call to return flattened observation instead of normal OrderedDict.

        Args:
            action (np.array): Action to take in environment
        Returns:
            4-tuple:

                - (np.array) flattened observations from the environment
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) misc information
        """
        ob_dict, reward, done, info = self.env.step(action)
        info['next_state'] = self._flatten_obs(ob_dict)
        self.state = info['next_state']
        return self._process_image(ob_dict), reward, done, info

    @staticmethod
    def _process_image(ob_dict):
        im = ob_dict['agentview_image']
        im = np.flip(im, axis=0)
        im = (resize(im, (64, 64)) * 255).astype(np.uint8)
        im = im.transpose((2, 0, 1))
        return im


# class NutAssemblyWrapper(gym.Wrapper):
#     def __init__(self, env):
#         super(NutAssemblyWrapper, self).__init__(env)
#         self.gripper_closed = False
#
#     def reset(self):
#         obs = super(NutAssemblyWrapper, self).reset()
#         self.gripper_closed = False
#         return obs
#
#     def step(self, action: np.ndarray):
#         next_obs, reward, done, info = super(NutAssemblyWrapper, self).step(action)
#         # repeat_action = np.zeros_like(action)
#         # repeat_action[-1] = action[-1]
#         # for _ in range(10):
#         #     if done:
#         #         break
#         #     next_obs, reward, done, info = super(NutAssemblyWrapper, self).step(action)
#         if action[-1] > 0:
#             self.gripper_closed = action[-1] > 0
#         else:
#             self.gripper_closed = False
#         return next_obs, reward, done, info


def get_config(env_name, camera_obs=False):
    controller_config = load_controller_config(default_controller='OSC_POSE')
    env_kwargs = {
        "env_name": env_name,
        "controller_configs": controller_config,
        "robots": [
            "UR5e" if env_name == 'NutAssembly' else 'Panda'
        ],
        "control_freq": 20,
        "hard_reset": False,
        "horizon": 200,
        "ignore_done": False,
        "reward_scale": 1.0,
        'has_renderer': False,
        'has_offscreen_renderer': camera_obs,
        'use_object_obs': True,
        'use_camera_obs': camera_obs,
        'reward_shaping': False,
        'render_camera': "agentview",
        'keys': [
            'object-state', 'robot0_proprio-state',
        ]
    }
    if env_name == 'NutAssembly':
        env_kwargs['nut_type'] = 'round'
        env_kwargs['single_object_mode'] = 2
    if env_name == 'TwoArmPegInHole':
        env_kwargs['robots'] = [
            'Panda',
            'Panda'
        ]
        env_kwargs['keys'] = [
            'object-state', 'robot0_proprio-state', 'robot1_proprio-state',
        ]
    return env_kwargs


def make_env(env_name, vis_dir=None):
    do_vis = vis_dir is not None
    env_kwargs = get_config(env_name, camera_obs=True)
    keys = env_kwargs.pop('keys')
    from_images = False

    env = suite.make(
        **env_kwargs
    )
    if from_images:
        env = RSGymWrapper(env, keys=keys)
        if do_vis:
            env = SimpleVideoSaver(env, video_dir=vis_dir)
    else:
        if do_vis:
            env = RSVisualizationGymWrapper(env, vis_dir, keys=keys)
        else:
            env = GymWrapper(env, keys)
    env = RSWrapper(env)
    # if env_name == 'NutAssembly':
    #     env = NutAssemblyWrapper(env)
    # env = NormalizedBoxEnv(env)
    # env = TerminateOnCompleteWrapper(env)

    return env


class NutAssemblySupervisor:  # move to its own file for better organization?
    # Algorithmic supervisor for nut assembly task
    def __init__(self):
        self.last_turn = None  # whether we last turned CW or CCW
        self.gripper_closed = False
        self.gripper_close_status = 0

    def __call__(self, obs):
        return self.act(obs)

    def act(self, obs):
        if self.gripper_close_status != 0:
            direction = self.gripper_close_status / np.abs(self.gripper_close_status)
            act = np.zeros(7)
            act[-1] = direction
            self.gripper_closed = act[-1] > 0
            self.gripper_close_status -= direction
            return act

        obj_pos, obj_quat = obs[:3], obs[3:7]
        rel_quat = obs[10:14]
        eef_pos, eef_quat = obs[32:35], obs[35:39]
        act = np.zeros(7)

        pose = pose2mat((obj_pos, obj_quat))
        grasp_point = (pose @ np.array([0.06, 0, 0, 1]))[:-1]

        # open and lift gripper if it's not holding anything.
        if self.gripper_closed and np.linalg.norm(grasp_point - eef_pos) > 0.02:
            act[-1] = -1.
            act[2] = 1.0
            self.gripper_closed = act[-1] > 0
            return act

        # move gripper to be aligned with washer handle.
        if not self.gripper_closed and np.linalg.norm(grasp_point[:2] - eef_pos[:2]) > 0.005:
            act[-1] = -1.
            act[0:2] = 50 * (grasp_point[:2] - eef_pos[:2])
            self.last_turn = None
            self.gripper_closed = act[-1] > 0
            return act

        # rotate gripper to be perpendicular to the washer.
        if not self.gripper_closed and abs(rel_quat[0] + 1) > 0.01 and abs(
                rel_quat[1] + 1) > 0.01:
            act[-1] = -1.
            if self.last_turn:
                act[5] = self.last_turn
            elif abs(rel_quat[0] + 1) < abs(rel_quat[1] + 1):  # rotate CW
                act[5] = -0.3
                self.last_turn = -0.3
            else:  # rotate CCW
                act[5] = 0.3
                self.last_turn = 0.3
            self.gripper_closed = act[-1] > 0
            return act

        # move gripper to the height of the washer.
        if not self.gripper_closed and abs(obj_pos[2] - eef_pos[2]) > 0.0075:
            act[-1] = -1.
            act[2] = 30 * (obj_pos[2] - eef_pos[2])
            self.gripper_closed = act[-1] > 0
            return act

        # grasp washer.
        if not self.gripper_closed:
            act[-1] = 1.
            self.gripper_close_status = 9
            self.gripper_closed = act[-1] > 0
            return act

        # move washer to correct height.
        cylinder_pos = np.array([0.22690132, -0.10067187, 1.0])
        if np.linalg.norm(cylinder_pos[:2] - obj_pos[:2]) > 0.005 and abs(
                cylinder_pos[2] - eef_pos[2]) > 0.01:
            act[-1] = 1.
            target_height = 1.0
            act[2] = 50 * (cylinder_pos[2] - eef_pos[2])
            self.gripper_closed = act[-1] > 0
            return act

        # center above the cylinder.
        if np.linalg.norm(cylinder_pos[:2] - obj_pos[:2]) > 0.005:
            act[-1] = 1.
            act[0:2] = 50 * (cylinder_pos[:2] - obj_pos[:2])
            self.gripper_closed = act[-1] > 0
            return act

        # lower washer down the cylinder.
        act[-1] = 1.
        act[2] = 50 * (0.83 - eef_pos[2])
        self.gripper_closed = act[-1] > 0
        return act
