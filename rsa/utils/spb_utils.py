from ctypes import Union

import rsa.envs.simple_point_bot as spb
from rsa.algos import MCSAC, MCTD3
# from rsa.algos import DDPGRSA
import rsa.utils.pytorch_utils as ptu

import numpy as np
from tqdm import tqdm
import torch


def plot_Q(policy,
           env: spb.SimplePointBot,
           file=None,
           plot=True,
           show=False,
           points=None,
           skip=10):
    data = np.zeros((env.window_height, env.window_width))
    for y in range(0, env.window_height, skip):
        row_states = []
        for x in range(0, env.window_width, skip):
            state = np.divide((x, y), (env.window_width, env.window_height))
            row_states.append(state)
        row_states = ptu.torchify(row_states)
        if type(policy) == MCSAC:
            acts, _, _ = policy.policy.sample(row_states)
        elif type(policy) == MCTD3:
            acts = policy.actor(row_states)
        else:
            assert False, "wtf"
        acts = acts.squeeze()
        q = policy.critic.Q1(row_states, acts).cpu().detach().numpy().squeeze()
        # safety = policy.risk_critic.safety(row_states, acts)
        # vals = s_set.safe_set_probability_np(np.array(row_states)).squeeze()
        if skip == 1:
            data[y] = q.squeeze()
        else:
            for i in range(skip):
                for j in range(skip):
                    data[y + i, j::skip] = q

        # elif skip == 2:
        #     data[y, ::2], data[y, 1::2] = safety, safety,
        #     data[y + 1, ::2], data[y + 1, 1::2] = safety, safety
        # else:
        #     raise NotImplementedError("Albert has not implemented logic for skipping %d yet" % skip)

    data = np.maximum(data, -1500)

    if plot:
        env.draw(heatmap=data, points=points, file=file, show=show)

    return data


def plot_maxes(policy,
               env: spb.SimplePointBot,
               file=None,
               plot=True,
               show=False):
    def foo(bar):
        if len(bar) > 0:
            # print(np.array(bar).shape)
            return np.array(list(bar)) * (env.window_width, env.window_height)
        return None
    drtg_points = foo(policy.drtg_buffer)
    bellman_points = foo(policy.bellman_buffer)

    if plot:
        env.draw(points=drtg_points, points2=bellman_points, file=file, show=show)
