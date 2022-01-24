import numpy as np
import scipy.signal
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class Actor(nn.Module):
    def __init__(self, d_obs, d_act, max_action, ensemble_size=2):
        super(Actor, self).__init__()

        # for consistency purposes with original td3, ensemble size=2 means one actor, ensemble size>2 means ensemble_size actors
        self.ensemble_size = ensemble_size
        self.ensemble = nn.ModuleList()

        for _ in range(ensemble_size):
            a = nn.Sequential(
                nn.Linear(d_obs[0], 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, d_act[0]),
                nn.Tanh()
                )
            self.ensemble.append(a)

        self.max_action = max_action

    def forward(self, state):
        # a = F.relu(self.l1(state))
        # a = F.relu(self.l2(a))
        # if self.ensemble_size == 1:
        #     return self.max_action * self.ensemble[0](state)
        return torch.stack([self.max_action * actor(state) for actor in self.ensemble])

    def sample(self, state):
        actions = self(state)
        log_prob = torch.zeros_like(actions)
        return actions, log_prob, actions


class Critic(nn.Module):
    def __init__(self, d_obs, d_act, ensemble_size=2):
        super(Critic, self).__init__()

        self.ensemble = nn.ModuleList()

        for _ in range(ensemble_size):
            q = nn.Sequential(
                nn.Linear(d_obs[0] + d_act[0], 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            )
            self.ensemble.append(q)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        return tuple([q(sa) for q in self.ensemble])

    def variance(self, state, action):
        if len(state.shape) == 1:
            state = state[None]
            action = action[None]
        qf = self(state, action)
        variance = torch.var(torch.cat(qf), dim=0).squeeze()
        return variance

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        return self.ensemble[0](sa)


class MLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs):
        # Return output from network scaled to action space limits
        return self.act_limit * self.pi(obs)


class Ensemble(nn.Module):
    # Multiple policies
    def __init__(self, observation_space, action_space, device, hidden_sizes=(256,256),
                 activation=nn.ReLU, num_nets=5):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]
        self.num_nets = num_nets
        self.device = device
        # build policy and value functions
        self.pis = [MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit).to(device) for _ in range(num_nets)]
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation).to(device)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation).to(device)

    def act(self, obs, i=-1):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            if i >= 0: # optionally, only use one of the nets.
                return self.pis[i](obs).cpu().numpy()
            vals = list()
            for pi in self.pis:
                vals.append(pi(obs).cpu().numpy())
            return np.mean(np.array(vals), axis=0)

    def variance(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            vals = list()
            for pi in self.pis:
                vals.append(pi(obs).cpu().numpy())
            return np.square(np.std(np.array(vals), axis=0)).mean()

    def safety(self, obs, act):
        # closer to 1 indicates more safe.
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        act = torch.as_tensor(act, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return float(torch.min(self.q1(obs, act), self.q2(obs,act)).cpu().numpy())


class GaussianPolicy(nn.Module):
    def __init__(self, d_obs, d_act, hidden_dim, max_action=1):
        super(GaussianPolicy, self).__init__()

        self.linear1 = nn.Linear(d_obs[0], hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, d_act[0])
        self.log_std_linear = nn.Linear(hidden_dim, d_act[0])

        self.apply(weights_init_)

        # action rescaling
        # TODO: correct this if we have to deal with action spaces that aren't centered around 0
        self.action_scale = torch.FloatTensor((max_action,))
        self.action_bias = torch.FloatTensor((0.,))

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # print('------------')
        # print(log_prob.mean())
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        # print(log_prob.mean())
        log_prob = log_prob.sum(1, keepdim=True)
        # print(log_prob.mean())
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    def __init__(self, d_obs, d_act, hidden_dim, max_action=1):
        super(DeterministicPolicy, self).__init__()

        self.linear1 = nn.Linear(d_obs[0], hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, d_act[0])
        self.noise = torch.Tensor(d_act[0])

        self.apply(weights_init_)

        # action rescaling
        # TODO: correct this if we have to deal with action spaces that aren't centered around 0
        self.action_scale = torch.FloatTensor(max_action)
        self.action_bias = torch.FloatTensor(0.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation, nn.Sigmoid)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


# Q_risk network architecture for image observations
class CNNQFunction(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super(CNNQFunction, self).__init__()
        # Process the state
        self.conv1 = nn.Conv2d(obs_dim[-1],
                               128,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               bias=True)
        self.conv2 = nn.Conv2d(128,
                               64,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               bias=True)
        self.conv3 = nn.Conv2d(64,
                               16,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               bias=True)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(16)
        self.demo_bn1 = nn.BatchNorm2d(128)
        self.demo_bn2 = nn.BatchNorm2d(64)
        self.demo_bn3 = nn.BatchNorm2d(16)

        self.final_linear = nn.Linear(self.final_linear_size, hidden_dim)

        # Process the action
        self.linear_act1 = nn.Linear(act_dim, hidden_dim)
        self.linear_act2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_act3 = nn.Linear(hidden_dim, hidden_dim)

        # Q1 architecture

        # Post state-action merge
        self.linear1_1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.linear2_1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3_1 = nn.Linear(hidden_dim, 1)

        # Post state-action merge
        self.linear1_2 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.linear2_2 = nn.Linear(hidden_dim, hidden_dim)

        self.linear3_2 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        # Process the state
        bn1, bn2, bn3 = self.bn1, self.bn2, self.bn3

        conv1 = F.relu(bn1(self.conv1(state)))
        conv2 = F.relu(bn2(self.conv2(conv1)))
        conv3 = F.relu(bn3(self.conv3(conv2)))
        final_conv = conv3.view(-1, self.final_linear_size)

        final_conv = F.relu(self.final_linear(final_conv))

        # Process the action
        x0 = F.relu(self.linear_act1(action))
        x0 = F.relu(self.linear_act2(x0))
        x0 = self.linear_act3(x0)

        # Concat
        xu = torch.cat([final_conv, x0], 1)

        # Apply a few more FC layers in two branches
        x1 = F.relu(self.linear1_1(xu))
        x1 = F.relu(self.linear2_1(x1))

        x1 = F.sigmoid(self.linear3_1(x1))

        x2 = F.relu(self.linear1_2(xu))
        x2 = F.relu(self.linear2_2(x2))

        x2 = F.sigmoid(self.linear3_2(x2))
        return x1, x2
