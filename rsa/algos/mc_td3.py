import rsa.utils.pytorch_utils as ptu

import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import os
import numpy as np


class MCTD3:
    def __init__(self, params):

        self.max_action = params['max_action']
        self.discount = params['discount']
        self.tau = params['tau']
        self.policy_noise = params['policy_noise']
        self.noise_clip = params['noise_clip']
        self.policy_freq = params['policy_freq']
        self.batch_size = params['batch_size']
        self.batch_size_demonstrator = params['batch_size_demonstrator']
        self.bc_weight = params['bc_weight']
        self.bc_decay = params['bc_decay']
        self.do_drtg_bonus = params['do_drtg_bonus']
        self.plot_drtg_maxes = params['plot_drtg_maxes']
        self.do_bc_loss = params['do_bc_loss']
        self.do_q_filter = params['do_q_filter']

        self.total_it = 0
        self.running_risk = 1

        self.actor = Actor(params['d_obs'], params['d_act'], self.max_action).to(ptu.TORCH_DEVICE)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=params['lr_actor'])

        self.critic = Critic(params['d_obs'], params['d_act']).to(ptu.TORCH_DEVICE)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=params['lr_critic'])

        self.drtg_buffer = set()
        self.bellman_buffer = set()

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(ptu.TORCH_DEVICE)
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self, replay_buffer):
        # Sample from replay buffer
        out_dict = replay_buffer.sample(self.batch_size)
        obs, action, next_obs, reward, mask, drtg, expert = out_dict['obs'], out_dict['act'], \
                                                            out_dict['next_obs'], out_dict['rew'], \
                                                            out_dict['mask'], out_dict['drtg'], \
                                                            out_dict['expert']




        obs, action, next_obs, reward, mask, drtg, expert = \
            ptu.torchify(obs, action, next_obs, reward, mask, drtg, expert)


        info = {}

        ###############################################################################################

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise)\
                .clamp(-self.noise_clip, self.noise_clip)

            next_action = (self.actor_target(next_obs) + noise)\
                .clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_Q = torch.min(target_Q1, target_Q2).squeeze()
            target_Q = reward + mask * self.discount * target_Q
            target_Q = target_Q.squeeze()

            if self.do_drtg_bonus:
                if self.plot_drtg_maxes:
                    # print(len(self.drtg_buffer), len(self.bellman_buffer))
                    obs_np = ptu.numpify(obs)
                    nqv_np = ptu.numpify(target_Q)
                    drtg_np = ptu.numpify(drtg)
                    drtg_bigger = drtg_np > nqv_np
                    # print(nqv_np[np.logical_not(drtg_bigger)].shape, drtg_np[drtg_bigger])
                    self.drtg_buffer.update([tuple(x) for x in obs_np[drtg_bigger]])
                    self.bellman_buffer.update([tuple(x) for x in obs_np[np.logical_not(drtg_bigger)]])
                    print(len(self.drtg_buffer), len(self.bellman_buffer))

                target_Q = torch.max(target_Q, drtg)

        # print('--------------------------------------------------')
        # print(drtg)
        # print(reward)
        # print(target_Q)

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        current_Q1, current_Q2 = current_Q1.squeeze(), current_Q2.squeeze()

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        info['critic_loss'] = critic_loss.item()
        info['Q1'] = current_Q1.mean().item()
        info['Q2'] = current_Q2.mean().item()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            actor_q_loss = -self.critic.Q1(obs, self.actor(obs)).mean()

            # Behavior cloning auxiliary loss, inspired by DDPGfD paper
            if self.do_bc_loss:
                # Sample expert actions from the replay buffer
                out_dict = replay_buffer.sample_positive(self.batch_size_demonstrator, 'expert')
                obs, action = out_dict['obs'], out_dict['act']
                obs, action = ptu.torchify(obs, action)

                # Calculate loss as negative log prob of actions
                act_hat = self.actor(obs)
                losses = F.mse_loss(act_hat, action, reduction='none')

                # Optional Q filter
                if self.do_q_filter:
                    with torch.no_grad():
                        q_agent = self.critic.Q1(obs, act_hat)
                        q_expert = self.critic.Q1(obs, action)
                        q_filter = torch.gt(q_expert, q_agent).float()

                    if torch.sum(q_filter) > 0:
                        bc_loss = torch.sum(losses * q_filter) / torch.sum(q_filter)
                    else:
                        bc_loss = actor_q_loss * 0
                else:
                    bc_loss = torch.mean(losses)

            else:
                bc_loss = actor_q_loss * 0

            lambda_bc = self.bc_decay ** self.total_it * self.bc_weight
            actor_loss = actor_q_loss + lambda_bc * bc_loss

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            info['actor_loss'] = actor_loss.item()
            info['actor_q_loss'] = actor_q_loss.item()
            info['actor_bc_loss'] = bc_loss.item()

            # Update the frozen target models
            ptu.soft_update(self.critic, self.critic_target, 1 - self.tau)
            ptu.soft_update(self.actor, self.actor_target, 1 - self.tau)


        ############################################################################################################

        # Compute targets using bellman backup and target function
        # with torch.no_grad():
        #     next_state_action, next_state_log_pi, _ = self.policy.sample(next_obs)
        #     qf_list_next_target = self.critic_target(next_obs, next_state_action)
        #     min_qf_next_target = torch.min(torch.cat(qf_list_next_target, dim=1), dim=1)[0] \
        #                          - self.alpha * next_state_log_pi.squeeze()
        #     min_qf_next_target = min_qf_next_target.squeeze()
        #     next_q_value = reward + mask * self.discount * min_qf_next_target
        #
        #     if self.do_drtg_bonus:
        #         if self.plot_drtg_maxes:
        #             # print(len(self.drtg_buffer), len(self.bellman_buffer))
        #             obs_np = ptu.numpify(obs)
        #             nqv_np = ptu.numpify(next_q_value)
        #             drtg_np = ptu.numpify(drtg)
        #             drtg_bigger = drtg_np > nqv_np
        #             # print(nqv_np[np.logical_not(drtg_bigger)].shape, drtg_np[drtg_bigger])
        #             self.drtg_buffer.update([tuple(x) for x in obs_np[drtg_bigger]])
        #             self.bellman_buffer.update([tuple(x) for x in obs_np[np.logical_not(drtg_bigger)]])
        #
        #         next_q_value = torch.max(next_q_value, drtg)
        #
        #
        #
        # # Compute Q losses
        # # Two Q-functions to mitigate positive bias in the policy improvement step
        # qf_list = self.critic(obs, action)
        # # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        # qf_losses = [
        #     F.mse_loss(qf.squeeze(), next_q_value)
        #     for qf in qf_list
        # ]
        # # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        # qf_loss = sum(qf_losses)
        #
        # # Q function backward pass
        # self.critic_optim.zero_grad()
        # qf_loss.backward()
        # self.critic_optim.step()
        #
        # # Sample from policy, compute minimum Q value of sampled action
        # pi, log_pi, _ = self.policy.sample(obs)
        # qf_list_pi = self.critic(obs, pi)
        # min_qf_pi = torch.min(torch.cat(qf_list_pi, dim=1), dim=1)[0]
        #
        # # Calculate policy loss
        # # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        # policy_q_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
        #
        # # Behavior cloning auxiliary loss, inspired by DDPGfD paper
        # if self.do_bc_loss:
        #     # Sample expert actions from the replay buffer
        #     out_dict = replay_buffer.sample_positive(self.batch_size_demonstrator, 'expert')
        #     obs, action = out_dict['obs'], out_dict['act']
        #     obs, action = ptu.torchify(obs, action)
        #
        #     # Calculate loss as negative log prob of actions
        #     mean, log_std = self.policy(obs)
        #     dist = torch.distributions.Normal(mean, torch.exp(log_std))
        #     losses = -torch.mean(dist.log_prob(action), dim=1)
        #
        #     # Optional Q filter
        #     if self.do_q_filter:
        #         with torch.no_grad():
        #             q_agent = self.critic.Q1(obs, dist.sample())
        #             q_expert = self.critic.Q1(obs, action)
        #             q_filter = torch.gt(q_expert, q_agent).float()
        #
        #         if torch.sum(q_filter) > 0:
        #             bc_loss = torch.sum(losses * q_filter) / torch.sum(q_filter)
        #         else:
        #             bc_loss = policy_q_loss * 0
        #     else:
        #         bc_loss = torch.mean(losses)
        #
        # else:
        #     bc_loss = policy_q_loss * 0
        #
        # # Lambda from DDPGfD paper I believe
        # lambda_bc = self.bc_decay ** self.total_it * self.bc_weight
        # policy_loss = policy_q_loss + lambda_bc * bc_loss
        #
        # # Policy backward pass
        # self.policy_optim.zero_grad()
        # policy_loss.backward()
        # self.policy_optim.step()
        #
        # # Automatic entropy tuning
        # if self.automatic_entropy_tuning:
        #     alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        #
        #     self.alpha_optim.zero_grad()
        #     alpha_loss.backward()
        #     self.alpha_optim.step()
        #
        #     self.alpha = self.log_alpha.exp()
        #     alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        # else:
        #     alpha_loss = torch.tensor(0.).to(ptu.TORCH_DEVICE)
        #     alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs
        #
        # # if init:
        # #     ptu.hard_update(self.critic, self.critic_target)
        # if self.total_it % self.target_update_interval == 0:
        #     ptu.soft_update(self.critic, self.critic_target, 1 - self.tau)

        self.total_it += 1
        return info

    def save(self, folder):
        os.makedirs(folder, exist_ok=True)

        torch.save(self.critic.state_dict(), os.path.join(folder, "critic.pth"))
        torch.save(self.critic_optimizer.state_dict(), os.path.join(folder, "critic_optimizer.pth"))

        torch.save(self.actor.state_dict(), os.path.join(folder, "actor.pth"))
        torch.save(self.actor_optimizer.state_dict(), os.path.join(folder, "actor_optimizer.pth"))

    def load(self, folder):
        self.critic.load_state_dict(torch.load(os.path.join(folder, "critic.pth")))
        self.critic_optimizer.load_state_dict(
            torch.load(os.path.join(folder, "critic_optimizer.pth")))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(os.path.join(folder, "actor.pth")))
        self.actor_optimizer.load_state_dict(
            torch.load(os.path.join(folder, "actor_optimizer.pth")))


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim[0], 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim[0])

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim[0] + action_dim[0], 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim[0] + action_dim[0], 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1
