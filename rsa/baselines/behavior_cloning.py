import rsa.utils.pytorch_utils as ptu
import rsa.algos.core as core

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import os
from tqdm import trange
import numpy as np


class BehaviorCloning:
    def __init__(self, env, logdir):
        self.d_obs = env.observation_space.shape
        self.d_act = env.action_space.shape
        self.ac_ub, self.ac_lb = env.action_space.high, env.action_space.low
        self.model = core.Actor(self.d_obs, self.d_act, self.ac_ub[0], ensemble_size=1).to(ptu.TORCH_DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

        self.logdir = logdir

    def act(self, obs):
        self.model.eval()

        obs = ptu.torchify(obs)
        # Expand batch dim if necessary
        if obs.shape == self.d_obs:
            obs = obs[None]

        act = self.model(obs).squeeze()
        act = ptu.numpify(act)

        act = np.clip(act, self.ac_lb, self.ac_ub)

        return act

    def train(self, replay_buffer, n_iters=10000):
        self.model.train()

        losses = []
        for i in trange(n_iters):
            out_dict = replay_buffer.sample(32)
            obs = ptu.torchify(out_dict['obs'])
            act = ptu.torchify(out_dict['act'])

            acts_pred = self.model(obs)
            loss = F.mse_loss(acts_pred, act)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())
        # pu.loss_plot(losses, os.path.join(self.logdir, 'loss.pdf'), 'BC Loss')
        # self.model.save(os.path.join(self.logdir, 'bc.pth'))

    def save(self, file):
        torch.save(self.model.state_dict(), file)

    def load(self, file):
        self.model.load_state_dict(torch.load(file))


class BCModel(nn.Module):
    def __init__(self, d_obs, d_act):
        super(BCModel, self).__init__()
        self.d_in = d_obs if len(d_obs) == 3 else (d_obs[0] * d_obs[1], d_obs[2], d_obs[3])
        self.model = nn.Sequential(
            nn.Conv2d(self.d_in[0], 24, 5, 2),
            nn.ELU(),
            nn.Conv2d(24, 36, 5, 2),
            nn.ELU(),
            nn.Conv2d(36, 48, 5, 2),
            nn.ELU(),
            nn.Conv2d(48, 64, 3, 1),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(64, 100),
            nn.ELU(),
            nn.Linear(100, 50),
            nn.ELU(),
            nn.Linear(50, 10),
            nn.ELU(),
            nn.Linear(10, d_act),
        )

    def forward(self, obs):
        if len(obs.shape) == 5:
            shape = obs.shape
            obs = obs.reshape(shape[0], shape[1] * shape[2], shape[3], shape[4])
        return self.model(obs)

    def loss(self, obs, act):
        acts_pred = self(obs)
        return F.mse_loss(acts_pred, act)

    def save(self, file):
        torch.save(self.state_dict(), file)

    def load(self, file):
        self.load_state_dict(torch.load(file))
