import rsa.utils as utils

import argparse
import numpy as np


def parse_args(td3_args=False, sac_args=False, thrifty_args=False, gail_args=False):
    parser = argparse.ArgumentParser()

    # Experiment arguments
    parser.add_argument('--exper-name', type=str, default=None,
                        help='Experiment name to be used for output directory')
    parser.add_argument('--env', type=str, default='spb',
                        help='environment name')
    parser.add_argument('--supervisor', type=int, default=0,
                        help='The index of the supervisor you would like to use')
    parser.add_argument('--expert-file', type=str, default='experts/spb.pth',
                        help='path to saved expert for algorithmic supervisors')
    parser.add_argument('--no-offline-data', action='store_true',
                        help='If true, will not load offline data')
    parser.add_argument('--seed', '-s', type=int, default=-1,
                        help='random seed')
    parser.add_argument('--device', type=int, default=0,
                        help='CUDA devide to use')
    parser.add_argument('--gen-data', action='store_true',
                        help='add flag if we need to generate data')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint from which to load models')

    # Training arguments
    parser.add_argument('--train-episodes', type=int, default=100,
                        help='Number of training episodes')
    parser.add_argument('--total-timesteps', type=int, default=int(1e5))
    parser.add_argument('--init-iters', type=int, default=0,
                        help='How many iterations of pretraining to run')
    parser.add_argument('--do-vis', action='store_true')
    parser.add_argument('--buffer-size', type=int, default=int(1e6),
                        help='Replay buffer size')
    parser.add_argument('--n-demos', type=int, default=20,
                        help='How many demos to collect/load')
    parser.add_argument('--update-n-steps', type=int, default=1,
                        help='How many gradient steps to take each timestep')
    parser.add_argument('--eval-freq', type=int, default=500,
                        help='How many interacts between evaluating policy')
    parser.add_argument('--num-eval-episodes', default=10, type=int,
                        help='How many episodes to evaluate over')
    parser.add_argument('--start-timesteps', default=0, type=int,
                        help='How many timesteps to use initial random policy')
    parser.add_argument('--reward-shift', type=float, default=0)
    parser.add_argument('--reward-scale', type=float, default=1)

    if sac_args:
        add_sac_args(parser)

    if td3_args:
        add_td3_args(parser)

    if gail_args:
        add_gail_args(parser)

    args = parser.parse_args()

    params = vars(args)

    params['logdir'] = utils.get_file_prefix(params)
    params['data_folder'] = utils.get_data_dir(params)

    # TODO: add
    add_env_info(params)

    return params


def add_sac_args(parser):
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--greedy-exp-eps', type=float, default=0.0,
                        help='Probability of choosing a random action during exploration')
    parser.add_argument('--discount', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')
    parser.add_argument('--automatic-entropy-tuning', action='store_true',
                        help='Automaically adjust α (default: False)')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--batch-size-demonstrator', type=int, default=128,
                        help='Batch size for demonstrator BC loss, N_D in overcoming exploration paper')
    parser.add_argument('--bc-weight', type=float, default=1,
                        help='weight for behavior cloning loss, recommended 1000/batch_size')
    parser.add_argument('--do-bc-loss', action='store_true', help='Whether or not to do a bc loss')
    parser.add_argument('--do-q-filter', action='store_true',
                        help='whether or not to use a q-filter for BC loss')
    parser.add_argument('--bc-decay', type=float, default=1)
    parser.add_argument('--weight-decay', type=float, default=5e-3,
                        help='Coefficient for weight regularization')
    parser.add_argument('--num-steps', type=int, default=1000001, metavar='N',
                        help='maximum number of steps (default: 1000000)')
    parser.add_argument('--hidden-size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--updates-per-step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--target-update-interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--target-bug', action='store_true',
                        help='whether to train policy with target Q function')
    parser.add_argument('--do-expert-q-boost', action='store_true')
    parser.add_argument('--do-drtg-bonus', action='store_true')
    parser.add_argument('--plot-drtg-maxes', action='store_true')
    parser.add_argument('--q-ensemble-size', type=int, default=2)


def add_td3_args(parser):
    parser.add_argument("--expl-noise", type=float, default=0.1)  # Std of Gaussian exploration noise
    parser.add_argument("--batch-size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument('--batch-size-demonstrator', type=int, default=128,
                        help='Batch size for demonstrator BC loss, N_D in overcoming exploration paper')
    parser.add_argument("--discount", type=float, default=0.99)  # Discount factor
    parser.add_argument("--tau", type=float, default=0.005)  # Target network update rate
    parser.add_argument("--policy-noise", type=float, default=0.2)  # Noise added to target policy during critic update
    parser.add_argument("--noise-clip", type=float, default=0.5)  # Range to clip target policy noise
    parser.add_argument("--policy-freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument('--lr-actor', type=float, default=3e-4)
    parser.add_argument('--lr-critic', type=float, default=3e-4)

    parser.add_argument('--bc-weight', type=float, default=1,
                        help='weight for behavior cloning loss, recommended 1000/batch_size')
    parser.add_argument('--do-bc-loss', action='store_true', help='Whether or not to do a bc loss')
    parser.add_argument('--do-q-filter', action='store_true',
                        help='whether or not to use a q-filter for BC loss')
    parser.add_argument('--bc-decay', type=float, default=1)
    parser.add_argument('--do-drtg-bonus', action='store_true')
    parser.add_argument('--plot-drtg-maxes', action='store_true')

# Left this in case we try this as a baseline
def add_gail_args(parser):
    parser.add_argument('--gail-iters', type=int, default=200)
    parser.add_argument('--gail-steps-per-iter', type=int, default=2000)
    parser.add_argument('--horizon', type=int, default=None)
    parser.add_argument('--lambda', type=float, default=1e-3)
    parser.add_argument('--gae-gamma', type=float, default=0.99)
    parser.add_argument('--gae-lambda', type=float, default=0.99)
    parser.add_argument('--epsilon', type=float, default=0.01)
    parser.add_argument('--max-kl', type=float, default=0.01)
    parser.add_argument('--cg-damping', type=float, default=0.1)
    parser.add_argument('--normalize-advantage', dest='normalize-advantage', action='store_true')
    parser.add_argument('--no-normalize-advantage', dest='normalize-advantage', action='store_false')
    parser.set_defaults(normalize_advantage=True)
    # "num_iters": 500,
    # "num_steps_per_iter": 2000,
    # "horizon": null,
    # "lambda": 1e-3,
    # "gae_gamma": 0.99,
    # "gae_lambda": 0.99,
    # "epsilon": 0.01,
    # "max_kl": 0.01,
    # "cg_damping": 0.1,
    # "normalize_advantage": true


def add_env_info(params):
    params['from_images'] = False
    params['max_action'] = 1
    # from rsa.envs.probe_envs import probes
    # if params['env'] == 'spb':
    #     params['max_action'] = 1
    #     params['d_obs'] = np.array((2,))
    #     params['d_act'] = np.array((2,))
    # if params['env'] == 'hc':
    #     params['max_action'] = 1
    #     params['d_obs'] = np.array((17,))
    #     params['d_act'] = np.array((6,))
    # if params['env'] == 'NutAssembly':
    #     params['max_action'] = 1
    #     params['d_obs'] = np.array((51,))
    #     params['d_act'] = np.array((7,))
    # if params['env'] in probes:
    #     params['max_action'] = 1
    #     params['d_obs'] = np.array((1,))
    #     params['d_act'] = np.array((1,))
