"""
A robot that can exert force in cardinal directions. The robot's goal is to
reach the origin and it experiences zero-mean Gaussian Noise. State
representation is (x, y). Action representation is (dx, dy).
"""

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

# self.window_width = 180
# self.window_height = 150

MAX_FORCE = 3


class SimplePointBot(Env, utils.EzPickle):

    def __init__(self, from_pixels=True,
                 walls=None,
                 window_width=180,
                 window_height=150,
                 max_force=3,
                 start_pos=(30, 75),
                 end_pos=(150, 75),
                 horizon=100,
                 constr_penalty=-100,
                 goal_thresh=3,
                 noise_scale=0.125):
        utils.EzPickle.__init__(self)
        self.done = self.state = None
        self.horizon = horizon
        self.start = start_pos
        self.goal = end_pos
        self.goal_thresh = goal_thresh
        self.noise_scale = noise_scale
        self.constr_penalty = constr_penalty
        self.window_width = window_width
        self.window_height = window_height
        self.max_force = max_force
        self.action_space = Box(-np.ones(2), np.ones(2))
        self.observation_space = Box(-np.ones(2) * np.float('inf'), np.ones(2) * np.float('inf'))
        self._episode_steps = 0
        # self.obstacle = self._complex_obstacle(OBSTACLE_COORDS)
        if walls is None:
            walls = [((75, 55), (100, 95))]
        self.walls = [self._complex_obstacle(wall) for wall in walls]
        self.wall_coords = walls
        self._from_pixels = from_pixels
        self._image_cache = {}

    def step(self, a):
        a = self._process_action(a)
        old_state = self.state.copy()
        next_state = self._next_state(self.state, a)
        cur_reward = self.step_reward(self.state, a)
        self.state = next_state
        self._episode_steps += 1
        constr = self.obstacle(next_state)
        self.done = self._episode_steps >= self.horizon
        mask = 1
        if constr:
            self.done = True
            cur_reward = self.constr_penalty
            mask = 0
        if cur_reward == 0:
            self.done = True
            mask = 0

        obs = self.state / (self.window_width, self.window_height)
        return obs, cur_reward, self.done, {
            "constraint": constr,
            "reward": cur_reward,
            "state": old_state,
            "next_state": next_state,
            "action": a,
            'goal': cur_reward == 0,
            'mask': mask
        }

    def reset(self, random_start=False):
        if random_start:
            self.state = np.random.random(2) * (self.window_width, self.window_height)
            if self.obstacle(self.state):
                self.reset(True)
        else:
            self.state = self.start + np.random.randn(2)
        self.done = False
        self._episode_steps = 0
        obs = self.state / (self.window_width, self.window_height)
        return obs

    def render(self, mode='human'):
        return self._draw_state(self.state)

    def _next_state(self, s, a, override=False):
        if self.obstacle(s):
            return s

        next_state = s + a + self.noise_scale * np.random.randn(len(s))
        next_state = np.clip(next_state, (0, 0), (self.window_width, self.window_height))
        return next_state

    def step_reward(self, s, a):
        """
        Returns 1 if in goal otherwise 0
        """
        return int(np.linalg.norm(np.subtract(self.goal, s)) < self.goal_thresh) - 1

    def obstacle(self, s):
        return any([wall(s) for wall in self.walls])

    @staticmethod
    def _complex_obstacle(bounds):
        """
        Returns a function that returns true if a given state is within the
        bounds and false otherwise
        :param bounds: bounds in form [[X_min, Y_min], [X_max, Y_max]]
        :return: function described above
        """
        min_x, min_y = bounds[0]
        max_x, max_y = bounds[1]

        def obstacle(state):
            if type(state) == np.ndarray:
                lower = (min_x, min_y)
                upper = (max_x, max_y)
                state = np.array(state)
                component_viol = (state > lower) * (state < upper)
                return np.product(component_viol, axis=-1)
            if type(state) == torch.Tensor:
                lower = torch.from_numpy(np.array((min_x, min_y)))
                upper = torch.from_numpy(np.array((max_x, max_y)))
                component_viol = (state > lower) * (state < upper)
                return torch.prod(component_viol, dim=-1)

        return obstacle

    @staticmethod
    def _process_action(a):
        return np.clip(a, -1, 1) * MAX_FORCE

    def draw(self, trajectories=None, heatmap=None, points=None, points2=None, plot_starts=False, board=True,
             file=None,
             show=False):
        """
        Draws the desired trajectories and heatmaps (probably would be a safe set) to pyplot
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)

        if heatmap is not None:
            assert heatmap.shape == (self.window_height, self.window_width)
            # heatmap = np.flip(heatmap, axis=0)
            im = plt.imshow(heatmap, cmap='hot')
            plt.colorbar(im)

        if board:
            self.draw_board(ax)

        if trajectories is not None and type(trajectories) == list:
            if type(trajectories[0]) == list:
                self.plot_trajectories(ax, trajectories, plot_starts)
            if type(trajectories[0]) == dict:
                self.plot_trajectory(ax, trajectories, plot_starts)

        if points is not None:
            # print('YOOOO', len(points), points.max(), points.min())
            plt.scatter(points[:, 0], points[:, 1], marker=',', s=1, linewidths=0.1, c='tab:red')
        if points2 is not None:
            # print('YOOOO', len(points), points.max(), points.min())
            plt.scatter(points2[:, 0], points2[:, 1], marker=',', linewidths=0.1, s=1, color='tab:blue')

        ax.set_aspect('equal')
        ax.autoscale_view()
        plt.gca().invert_yaxis()

        if file is not None:
            plt.savefig(file)

        if show:
            plt.show()
        else:
            plt.close()

    def plot_trajectory(self, ax, trajectory, plot_start=False):
        self.plot_trajectories(ax, [trajectory], plot_start)

    def plot_trajectories(self, ax, trajectories, plot_start=False):
        """
        Renders a trajectory to pyplot. Assumes you already have a plot going
        :param ax:
        :param trajectories: Trajectories to impose upon the graph
        :param plot_start: whether or not to draw a circle at the start of the trajectory
        :return:
        """

        for trajectory in trajectories:
            states = np.array([frame['obs'] for frame in trajectory])
            plt.plot(states[:, 0], self.window_height - states[:, 1])
            if plot_start:
                start = states[0]
                start_circle = plt.Circle((start[0], self.window_height - start[1]), radius=2,
                                          color='lime')
                ax.add_patch(start_circle)

    def draw_board(self, ax):
        plt.xlim(0, self.window_width)
        plt.ylim(0, self.window_height)

        for wall in self.wall_coords:
            width, height = np.subtract(wall[1], wall[0])
            ax.add_patch(
                patches.Rectangle(
                    xy=wall[0],  # point of origin.
                    width=width,
                    height=height,
                    linewidth=1,
                    color='red',
                    fill=True
                )
            )

        circle = plt.Circle(self.start, radius=3, color='k')
        ax.add_patch(circle)
        circle = plt.Circle(self.goal, radius=3, color='k')
        ax.add_patch(circle)
        ax.annotate("start", xy=(self.start[0], self.start[1] - 8), fontsize=10,
                    ha="center")
        ax.annotate("goal", xy=(self.goal[0], self.goal[1] - 8), fontsize=10, ha="center")


class SPBVisWrapper(Wrapper):
    def __init__(self, env, vis_dir):
        super(SPBVisWrapper, self).__init__(env)
        self.vis_dir = vis_dir

        if not os.path.exists(self.vis_dir):
            os.makedirs(self.vis_dir)
        self.traj_buffer = []
        self.count = 0

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        next_obs_plot = next_obs[:2]
        self.traj_buffer.append(next_obs_plot * (self.window_width, self.window_height))
        return next_obs, reward, done, info

    def reset(self, **kwargs):
        if len(self.traj_buffer) > 0:
            self._draw()
        self.traj_buffer = []
        self.count += 1

        obs = self.env.reset(**kwargs)
        obs_plot = obs[:2]
        self.traj_buffer.append(obs_plot * (self.window_width, self.window_height))

        return obs

    def _draw(self):
        """
        Draws the desired trajectories and heatmaps (probably would be a safe set) to pyplot
        """
        file = os.path.join(self.vis_dir, '%d.pdf' % self.count)
        traj = np.array(self.traj_buffer)

        fig = plt.figure()
        ax = fig.add_subplot(111)

        plt.xlim(0, self.window_width)
        plt.ylim(0, self.window_height)

        if self.walls is not None:
            for wall in self.env.wall_coords:
                width, height = np.subtract(wall[1], wall[0])
                ax.add_patch(
                    patches.Rectangle(
                        xy=wall[0],  # point of origin.
                        width=width,
                        height=height,
                        linewidth=1,
                        color='red',
                        fill=True
                    )
                )

        if self.start is not None:
            circle = plt.Circle(self.start, radius=3, color='k')
            ax.add_patch(circle)
            ax.annotate("start", xy=(self.start[0], self.start[1] - 8), fontsize=10,
                        ha="center")
        if self.goal is not None:
            circle = plt.Circle(self.goal, radius=3, color='k')
            ax.add_patch(circle)
            ax.annotate("goal", xy=(self.goal[0], self.goal[1] - 8), fontsize=10, ha="center")

        # states = np.array([frame['obs'] for frame in trajectory])
        plt.plot(traj[:, 0], traj[:, 1])
        plt.gca().invert_yaxis()

        ax.set_aspect('equal')
        ax.autoscale_view()

        plt.savefig(file)
        plt.close()


def expert_pol_0(obs):
    obs = obs * (180, 150)
    x, y = obs
    if x < 140:
        if y > 35:
            goal = np.array((30, 15))
        else:
            goal = np.array((150, 15))
    else:
        goal = np.array((150, 75))

    act = np.subtract(goal, obs) / 2
    act = np.clip(act, -1, 1)
    return act


def expert_pol_1(obs):
    """Expert policy for default pointbot"""
    obs = obs * (180, 150)
    obs = obs.reshape((-1, 2))
    # These are boolean arrays for whether its state indicates that it is in the first or
    # second half of the trajectory
    first_half = obs[:, 0] < 180 // 2
    second_half = 1 - first_half
    # An array of actions taking it up and to the right
    up_right = np.array(((1, -1),))
    home = np.subtract((150, 75), obs)
    home = np.clip(home, -1 / 2, 1 / 2)
    return (up_right.T * first_half.T + second_half.T * home.T).T.squeeze()


expert_pols = [
    expert_pol_0,
    expert_pol_1
]


class MediumPointBot(SimplePointBot):
    def __init__(self):
        super(MediumPointBot, self).__init__(from_pixels=False,
                                             window_width=180,
                                             window_height=150,
                                             start_pos=(20, 75),
                                             end_pos=(160, 75),
                                             walls=[
                                                 ((80, 0), (100, 40)),
                                                 ((80, 45), (100, 150)),
                                             ],
                                             horizon=100,
                                             constr_penalty=-100)

    # def step(self, a):
    #     a = self._process_action(a)
    #     old_state = self.state.copy()
    #     next_state = self._next_state(self.state, a)
    #     cur_reward = self.step_reward(self.state, a)
    #     self._episode_steps += 1
    #     constr = self.obstacle(next_state)
    #     self.done = self._episode_steps >= self.horizon
    #     mask = 1
    #     if constr:
    #         next_state = old_state
    #     if cur_reward == 0:
    #         self.done = True
    #         mask = 0
    #
    #     self.state = next_state
    #
    #     obs = self.state / (self.window_width, self.window_height)
    #     return obs, cur_reward, self.done, {
    #         "constraint": 0,
    #         "reward": cur_reward,
    #         "state": old_state,
    #         "next_state": next_state,
    #         "action": a,
    #         'goal': cur_reward == 0,
    #         'mask': mask
    #     }


def mpb_expert(obs):
    obs = obs * (180, 150)
    x, y = obs
    if x < 101:
        if 40 < y < 45:
            goal = (105, 42.5)
        else:
            goal = (78, 42.5)
    else:
        goal = (160, 75)

    act = np.subtract(goal, obs) / 2
    act = act / np.max(np.abs(act))
    return act


class HardPointBot(SimplePointBot):
    def __init__(self):
        super(HardPointBot, self).__init__(from_pixels=False,
                                           window_width=180,
                                           window_height=150,
                                           start_pos=(20, 75),
                                           end_pos=(160, 75),
                                           walls=[
                                               ((55, 0), (75, 40)),
                                               ((55, 45), (75, 150)),
                                               ((105, 0), (125, 100)),
                                               ((105, 105), (125, 150))
                                           ],
                                           horizon=100,
                                           constr_penalty=-100)

    def step(self, a):
        a = self._process_action(a)
        old_state = self.state.copy()
        next_state = self._next_state(self.state, a)
        cur_reward = self.step_reward(self.state, a)
        self._episode_steps += 1
        constr = self.obstacle(next_state)
        self.done = self._episode_steps >= self.horizon
        mask = 1
        if constr:
            next_state = old_state
        if cur_reward == 0:
            self.done = True
            mask = 0

        self.state = next_state

        obs = self.state / (self.window_width, self.window_height)
        return obs, cur_reward, self.done, {
            "constraint": 0,
            "reward": cur_reward,
            "state": old_state,
            "next_state": next_state,
            "action": a,
            'goal': cur_reward == 0,
            'mask': mask
        }


def hpb_expert(obs):
    obs = obs * (180, 150)
    x, y = obs
    if x < 76:
        if 40 < y < 45:
            goal = (80, 42.5)
        else:
            goal = (53, 42.5)
    elif x < 126:
        if 100 < y < 105:
            goal = (126, 102.5)
        else:
            goal = (103, 102.5)
    else:
        goal = (160, 75)

    act = np.subtract(goal, obs) / 2
    act = act / np.max(np.abs(act))
    return act


class SimplePointBotExtraLongEasy(SimplePointBot):
    def __init__(self, from_pixels=False):
        super().__init__(from_pixels,
                         window_width=1000,
                         window_height=800,
                         start_pos=(250, 100),
                         end_pos=(750, 100),
                         walls=[
                             ((490, 0), (510, 600)),
                             ((490, 610), (510, 800)),
                             ((0, 500), (100, 520)),
                             ((110, 500), (500, 520)),
                             ((500, 200), (900, 220)),
                             ((910, 200), (1000, 220)),
                         ],
                         horizon=1000,
                         constr_penalty=-1000)

    def step(self, a):
        a = self._process_action(a)
        old_state = self.state.copy()
        next_state = self._next_state(self.state, a)
        cur_reward = self.step_reward(self.state, a)
        self._episode_steps += 1
        constr = self.obstacle(next_state)
        self.done = self._episode_steps >= self.horizon
        mask = 1
        if constr:
            next_state = old_state
        if cur_reward == 0:
            self.done = True
            mask = 0

        self.state = next_state

        obs = self.state / (self.window_width, self.window_height)
        return obs, cur_reward, self.done, {
            "constraint": 0,
            "reward": cur_reward,
            "state": old_state,
            "next_state": next_state,
            "action": a,
            'goal': cur_reward == 0,
            'mask': mask
        }


class SimplePointBotExtraLong(SimplePointBot):
    def __init__(self, from_pixels=True):
        super().__init__(from_pixels,
                         window_width=1000,
                         window_height=800,
                         start_pos=(250, 100),
                         end_pos=(750, 100),
                         walls=[
                             ((490, 0), (510, 600)),
                             ((490, 610), (510, 800)),
                             ((0, 500), (100, 520)),
                             ((110, 500), (500, 520)),
                             ((500, 200), (900, 220)),
                             ((910, 200), (1000, 220)),
                         ],
                         horizon=600,
                         constr_penalty=-600)


def spbxl_expert(obs):
    obs = obs * (1000, 800)
    x, y = obs
    if x < 515:
        if y < 525:
            if 100 < x < 110:
                goal = (105, 530)
            else:
                goal = (105, 490)
        else:
            if 600 < y < 610:
                goal = (525, 605)
            else:
                goal = (495, 605)
    else:
        if y > 195:
            if 900 < x < 910:
                goal = (905, 190)
            else:
                goal = (905, 225)
        else:
            goal = (750, 100)

    act = np.subtract(goal, obs) / 2
    act = np.clip(act, -1, 1)
    return act


class SimplePointBotLong(SimplePointBot):
    def __init__(self, from_pixels=True):
        super().__init__(from_pixels,
                         start_pos=(15, 20),
                         end_pos=(165, 20),
                         walls=[((80, 55), (100, 150)),
                                ((30, 0), (45, 100)),
                                ((30, 120), (45, 150)),
                                ((135, 0), (150, 120))],
                         horizon=500)
