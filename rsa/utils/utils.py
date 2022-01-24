# import rsa.utils.robosuite_utils as ru
import rsa.envs.simple_point_bot as spb
import rsa.envs.random_point_bot as rpb
from rsa.envs.probe_envs import Probe1, Probe2, probes
from rsa.envs import SimpleVideoSaver, Push, ObjExtraction
from rsa.utils.sac_supervisor import SacSupervisor
import rsa.utils.pytorch_utils as ptu
from rsa.utils.replay_buffer import ReplayBuffer

import os
import torch
import numpy as np
import random
import gym
from gym.wrappers import TimeLimit
from datetime import datetime
from collections.abc import Iterable
import json
import logging

import d4rl

log = logging.getLogger("utils")

d4rl_envs = ('pen-human-v1', 'antmaze-medium-diverse-v0','door-human-v1', 'hammer-human-v1', 'relocate-human-v1')
# pointbot_envs = ('spb', 'rpb', 'lpb', 'hpb', 'mpb', 'lpb_easy')
pointbot_envs = {
    'spb': spb.SimplePointBot,
    'rpb': rpb.RandomPointBot,
    'lpb': spb.SimplePointBotExtraLong,
    'lpb_easy': spb.SimplePointBotExtraLongEasy,
    'hpb': spb.HardPointBot,
    'mpb': spb.MediumPointBot
}
mujoco_envs = ['Hopper-v3', 'HalfCheetah-v3', 'Ant-v3', 'Walker2d-v3', 'Humanoid-v3']


def seed(s, envs=None):
    if s == -1:
        return

    # torch.set_deterministic(True)
    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)

    if envs is not None:
        if isinstance(envs, Iterable):
            for env in envs:
                env.seed(s)
                env.action_space.seed(s)
        else:
            envs.seed(s)
            envs.action_space.seed(s)


def get_file_prefix(params=None):
    if params is not None and params['exper_name'] is not None:
        folder = os.path.join('outputs', params['exper_name'])
    else:
        now = datetime.now()
        date_string = now.strftime("%Y-%m-%d/%H-%M-%S")
        folder = os.path.join('outputs', date_string)
    if params is not None and params['seed'] != -1:
        folder = os.path.join(folder, str(params['seed']))
    return folder


def get_data_dir(params):
    return os.path.join('data', params['env'], str(params['supervisor']))


def init_logging(folder, file_level=logging.INFO, console_level=logging.DEBUG):
    # set up logging to file
    logging.basicConfig(level=file_level,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M:%S',
                        filename=os.path.join(folder, 'log.txt'),
                        filemode='w')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(console_level)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)


def colorize(string, color, bold=False, highlight=False):
    """
    Colorize a string.

    This function was originally written by John Schulman.
    """
    color2num = dict(
        gray=30,
        red=31,
        green=32,
        yellow=33,
        blue=34,
        magenta=35,
        cyan=36,
        white=37,
        crimson=38
    )

    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)


def make_env(params):
    env_name = params['env']
    use_robosuite = env_name in ('Lift', 'Door', 'NutAssembly', 'TwoArmPegInHole')

    # load expert/env here...
    # if use_robosuite:
    #     env = ru.make_env(params['env'])
    #     test_env = ru.make_env(params['env'], os.path.join(params['logdir'], 'vis'))
    if env_name in pointbot_envs:
        cls = pointbot_envs[env_name]
        env = spb.SPBVisWrapper(cls(),
                                os.path.join(params['logdir'], 'vis_train'))
        test_env = spb.SPBVisWrapper(cls(),
                                     os.path.join(params['logdir'], 'vis_test'))
    elif env_name == 'push':
        env = TimeLimit(Push(), max_episode_steps=150)
        test_env = TimeLimit(Push(), max_episode_steps=150)
        if params['do_vis']:
            test_env = SimpleVideoSaver(test_env,
                                        os.path.join(params['logdir'], 'vis_test'),
                                        from_render=True,
                                        camera='maincam')
    elif env_name == 'extraction':
        env = TimeLimit(ObjExtraction(), max_episode_steps=50)
        test_env = TimeLimit(ObjExtraction(), max_episode_steps=50)
        if params['do_vis']:
            test_env = SimpleVideoSaver(test_env,
                                        os.path.join(params['logdir'], 'vis_test'),
                                        from_render=True,
                                        camera='maincam')
    elif use_robosuite:
        env = ru.make_env(params['env'], os.path.join(params['logdir'], 'vis_train'))
        test_env = ru.make_env(params['env'], os.path.join(params['logdir'], 'vis_test'))
    else:
        env = gym.make(params['env'])
        test_env = gym.make(params['env'])
        if params['do_vis']:
            speedup = 4 if env_name in ('Hopper-v3', 'Walker2d-v3') else 1
            camera = 'track' if env_name in mujoco_envs else None
            test_env = SimpleVideoSaver(test_env,
                                        os.path.join(params['logdir'], 'vis_test'),
                                        from_render=True,
                                        speedup=speedup,
                                        camera=camera)

#    if params['env'] in d4rl_envs:
 #       env = D4RLWrapper(env)
  #      test_env = D4RLWrapper(test_env)

    if env_name in pointbot_envs:
        params['horizon'] = env.horizon
    elif hasattr(env, '_max_episode_steps'):
        params['horizon'] = env._max_episode_steps
    else:
        # TODO fix this lmao
        params['horizon'] = 300

    params['d_obs'] = env.observation_space.shape
    params['d_act'] = env.action_space.shape

    return env, test_env


def make_expert_policy(params, env):
    env_name = params['env']
    use_robosuite = env_name in ('Lift', 'Door', 'NutAssembly', 'TwoArmPegInHole')
    # if use_robosuite:
    #     expert = SacSupervisor(env.observation_space.shape[0], env.action_space.shape[0])
    #     expert.load_supervisor(params['expert'])
    #     expert = expert.to(ptu.TORCH_DEVICE)
    #     expert_pol = lambda obs: expert.get_action(obs, True)
    if env_name in pointbot_envs:
        expert_pol = {
            'spb': spb.expert_pols[params['supervisor']],
            'rpb': rpb.expert_pol,
            'lpb': spb.spbxl_expert,
            'lpb_easy': spb.spbxl_expert,
            'mpb': spb.mpb_expert,
            'hpb': spb.hpb_expert
        }
    elif env_name in mujoco_envs:
        from rsa.algos.mc_sac import MCSAC
        with open('experts/mujoco_params.json') as f:
            expert_params = json.load(f)
        expert_params['max_action'] = params['max_action']
        expert_params['d_obs'] = params['d_obs']
        expert_params['d_act'] = params['d_act']
        sac = MCSAC(expert_params)
        sac.load(os.path.join('experts', env_name))
        expert_pol = lambda obs: sac.select_action(obs, evaluate=True)
    elif env_name in ('push', 'extraction'):
        expert_pol = lambda _: env.unwrapped.expert_action()
    elif params['env'] in probes:
        expert_pol = lambda _: np.random.random(1) * 2 - 1
    elif use_robosuite:
        experts = {
            'Lift': 'lift.pkl',
            'Door': 'door.pkl',
            'TwoArmPegInHole': 'tapih.pkl'
        }
        expert = SacSupervisor(env.observation_space.shape[0], env.action_space.shape[0])
        expert.load_supervisor(os.path.join('supervisors', experts[params['env']]))
        expert = expert.to(ptu.TORCH_DEVICE)
        expert_pol = lambda obs: expert.get_action(obs, True)
    else:
        expert = torch.load(params['expert'], map_location=ptu.TORCH_DEVICE).to(ptu.TORCH_DEVICE)
        expert_pol = lambda o: expert.act(
            torch.as_tensor(o, dtype=torch.float32, device=ptu.TORCH_DEVICE))

    # expert_pol_clipped = lambda obs: np.clip(expert_pol(obs), -params['max_action'],
    #                                          params['max_action'])
    return expert_pol


def save_trajectory(trajectory, data_folder, i):
    # If the observations are images save them as separate numpy arrays
    do_image_filtering = len(trajectory[0]['obs'].shape) == 3
    if do_image_filtering:
        im_fields = ('obs', 'next_obs')
        for field in im_fields:
            if field in trajectory[0]:
                dat = np.array([frame[field] for frame in trajectory], dtype=np.uint8)
                np.save(os.path.join(data_folder, "%d_%s.npy" % (i, field)), dat)
        traj_save = [{key: frame[key] for key in frame if key not in im_fields}
                     for frame in trajectory]
    else:
        traj_save = trajectory

    for frame in traj_save:
        for key in frame:
            if type(frame[key]) == np.ndarray:
                # if frame[key].dtype == np.float64:
                #     frame[key] = frame[key].astype(np.float32)
                frame[key] = tuple(frame[key].tolist())

    with open(os.path.join(data_folder, "%d.json" % i), "w") as f:
        json.dump(traj_save, f)


def load_trajectory(data_folder, i):
    with open(os.path.join(data_folder, '%d.json' % i), 'r') as f:
        trajectory = json.load(f)

    # Add images stored as .npy files if there is no obs in the json
    add_images = 'obs' not in trajectory[0]
    if add_images:
        im_fields = ('obs', 'next_obs')
        im_dat = {}

        for field in im_fields:
            f = os.path.join(data_folder, "%d_%s.npy" % (i, field))
            if os.path.exists(data_folder):
                dat = np.load(f)
                im_dat[field] = dat.astype(np.uint8)

        for j, frame in list(enumerate(trajectory)):
            for key in im_dat:
                frame[key] = im_dat[key][j]

    return trajectory


def load_replay_buffer(params, add_drtg=False):
    replay_buffer = ReplayBuffer(params['buffer_size'])
    if not params['no_offline_data']:
        for i in range(params['n_demos']):
            trajectory = load_trajectory(params['data_folder'], i)
            if add_drtg:
                # print('--------------')
                x = shift_reward(trajectory[-1]['rew'], params) / (1 - params['discount'])
                for transition in reversed(trajectory):
                    # transition['rew'] += 1
                    # print(x)
                    transition['rew'] = shift_reward(transition['rew'], params)
                    x = transition['rew'] + transition['mask'] * params['discount'] * x
                    transition['drtg'] = x
            replay_buffer.store_trajectory(trajectory)

    return replay_buffer


def load_d4rl_replay_buffer(env, params, add_drtg=False):
    replay_buffer = ReplayBuffer(params['buffer_size'])
    if not params['no_offline_data']:
        data = env.get_dataset()
        obss = data['observations'][:-1]
        next_obss = data['observations'][1:]
        acts = data['actions'][:-1]
        rews = data['rewards'][:-1].astype(int)
        # rews = (data['rewards'][:-1] > 0) - 1
        timeouts = data['terminals'][:-1]

        drtg, done = 0, True
        for obs, act, next_obs, rew, timeout in reversed(
                list(zip(obss, acts, next_obss, rews, timeouts))):
            if timeout:
                drtg = 0
                done = True
                continue

            rew = shift_reward(rew, params)

            transition = {
                'obs': obs,
                'next_obs': next_obs,
                'act': act,
                'rew': rew,
                'mask': 1,
                'expert': 1,
                'done': done
            }
            if add_drtg:
                drtg = rew + params['discount'] * drtg
                transition['drtg'] = drtg

            replay_buffer.store_transition(transition)
            done = False

    return replay_buffer


def generate_offline_data(env, expert_policy, params):
    # Runs expert policy in the environment to collect data
    i = 0
    total_rews = []
    act_limit = env.action_space.high[0]
    try:
        os.makedirs(params['data_folder'])
    except FileExistsError:
        x = input(
            'Data already exists. Type `o` to overwrite, type anything else to skip data collection... > ')
        if x.lower() != 'o':
            return
    while i < params['n_demos']:
        print('Collecting demo %d' % i)
        obs, total_ret, done, t, completed = env.reset(), 0, False, 0, False
        trajectory = []
        while not done:
            act = expert_policy(obs)
            # act = act + np.random.normal(0, 0.3, (2,))
            if act is None:
                done, rew = True, 0
                continue
            # act = np.clip(act, -act_limit, act_limit)
            next_obs, rew, done, info = env.step(act)

            trajectory.append({
                'obs': obs,
                'next_obs': next_obs,
                'act': act.astype(np.float64),
                'rew': rew,
                'done': done,
                'mask': info['mask'] if 'mask' in info
                    else (1 if t + 1 == params['horizon'] else float(not done)),
                'expert': 1
            })
            # print(t, params['horizon'])

            total_ret += rew
            if 'goal' in info:
                completed = completed or info['goal']
            else:
                completed = True
            t += 1
            obs = next_obs

        # TODO: Currently it rejects all unsuccessful demo trajectories, reconsider in future
        if True:
            if completed:  # only count successful episodes
                save_trajectory(trajectory, params['data_folder'], i)
                i += 1
            else:
                print('Trajectory unsuccessful, redoing')
            env.close()
        # else:
        #     i += 1
        #     obss.extend(curr_obs)
        #     acts.extend(curr_acs)
        #     next_obss.extend(curr_next_obss)
        #     rews.extend(curr_rews)
        #     dones.extend(curr_dones)
        log.info("Collected episode with return {}".format(total_ret))
        total_rews.append(total_ret)
    log.info("Ep Mean, Std Dev:", np.array(total_rews).mean(), np.array(total_rews).std())


def shift_reward(rew, params):
    return (rew + params['reward_shift']) * params['reward_scale']


def add_dicts(*args):
    out = {}
    for arg in args:
        for k, v in arg.items():
            out[k] = v
    return out
