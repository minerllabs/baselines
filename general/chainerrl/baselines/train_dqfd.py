"""original source: https://github.com/chainer/chainerrl/pull/480

MIT License

Copyright (c) Preferred Networks, Inc.
"""

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *
from future import standard_library
standard_library.install_aliases()

import argparse
from inspect import getsourcefile
import os
import sys

import numpy as np

import chainer

import minerl  # noqa: register MineRL envs as Gym envs.
import gym
import chainerrl
from chainerrl import experiments, explorers
from chainerrl.experiments.evaluator import Evaluator
from dqfd import DQfD, PrioritizedDemoReplayBuffer

from q_functions import CNNBranchingQFunction
from env_wrappers import (
    BranchedRandomizedAction, BranchedActionWrapper,
    MoveAxisWrapper, FrameSkip, FrameStack, ObtainPoVWrapper,
    PoVWithCompassAngleWrapper, FullObservationSpaceWrapper)
from expert_converter import choose_top_experts, fill_buffer


class ScaleGradHook(object):
    name = 'ScaleGrad'
    call_for_each_param = True
    timing = 'pre'

    def __init__(self, scale):
        self.scale = scale

    def __call__(self, rule, param):
        if getattr(param, 'scale_param', False):
            param.grad *= self.scale


def main():
    """Parses arguments and runs the example
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='MineRLTreechop-v0',
                        choices=[
                            'MineRLTreechop-v0',
                            'MineRLNavigate-v0', 'MineRLNavigateDense-v0', 'MineRLNavigateExtreme-v0', 'MineRLNavigateExtremeDense-v0',
                            'MineRLObtainIronPickaxe-v0', 'MineRLObtainIronPickaxeDense-v0',
                            'MineRLObtainDiamond-v0', 'MineRLObtainDiamondDense-v0',
                            'MineRLNavigateDenseFixed-v0'  # for debug use
                        ],
                        help='MineRL environment identifier')
    parser.add_argument('--outdir', type=str, default='results',
                        help='Directory path to save output files.'
                             ' If it does not exist, it will be created.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed [0, 2 ** 31)')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='GPU to use, set to -1 if no GPU.')
    parser.add_argument('--final-exploration-frames',
                        type=int, default=10**6,
                        help='Timesteps after which we stop ' +
                        'annealing exploration rate')
    parser.add_argument('--final-epsilon', type=float, default=0.01,
                        help='Final value of epsilon during training.')
    parser.add_argument('--eval-epsilon', type=float, default=0.001,
                        help='Exploration epsilon used during eval episodes.')
    parser.add_argument('--replay-start-size', type=int, default=1000,
                        help='Minimum replay buffer size before ' +
                        'performing gradient updates.')
    parser.add_argument('--target-update-interval', type=int, default=10**4,
                        help='Frequency (in timesteps) at which ' +
                        'the target network is updated.')
    parser.add_argument('--update-interval', type=int, default=4,
                        help='Frequency (in timesteps) of network updates.')
    parser.add_argument('--eval-n-runs', type=int, default=10)
    parser.add_argument('--no-clip-delta',
                        dest='clip_delta', action='store_false')
    parser.add_argument('--error-max', type=float, default=1.0)
    parser.add_argument('--num-step-return', type=int, default=10)
    parser.set_defaults(clip_delta=True)
    parser.add_argument('--logging-level', type=int, default=20,
                        help='Logging level. 10:DEBUG, 20:INFO etc.')
    parser.add_argument('--logging-filename', type=str, default=None)
    parser.add_argument('--monitor', action='store_true', default=False,
                       help='Monitor env. Videos and additional information are saved as output files when evaluation')
    # parser.add_argument('--render', action='store_true', default=False,
    # help='Render env states in a GUI window.')
    parser.add_argument('--optimizer', type=str, default='rmsprop',
                        choices=['rmsprop', 'adam'])
    parser.add_argument('--lr', type=float, default=2.5e-4, help='Learning rate')
    parser.add_argument("--replay-buffer-size", type=int, default=10**6,
                        help="Size of replay buffer (Excluding demonstrations)")
    parser.add_argument("--minibatch-size", type=int, default=32)
    parser.add_argument('--batch-accumulator', type=str, default="sum")
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument("--save-demo-trajectories", action="store_true",
                        default=False)

    # DQfD specific parameters for loading and pretraining.
    parser.add_argument('--n-experts', type=int, default=10)
    parser.add_argument('--expert-demo-path', type=str, default=None)
    parser.add_argument('--n-pretrain-steps', type=int, default=750000)
    parser.add_argument('--demo-supervised-margin', type=float, default=0.8)
    parser.add_argument('--loss-coeff-l2', type=float, default=1e-5)
    parser.add_argument('--loss-coeff-nstep', type=float, default=1.0)
    parser.add_argument('--loss-coeff-supervised', type=float, default=1.0)
    parser.add_argument('--bonus-priority-agent', type=float, default=0.001)
    parser.add_argument('--bonus-priority-demo', type=float, default=1.0)

    # Action branching architecture
    parser.add_argument('--gradient-clipping', action='store_true', default=False)
    parser.add_argument('--gradient-rescaling', action='store_true', default=False)

    # NoisyNet parameters
    parser.add_argument('--use-noisy-net', type=str, default=None,
                        choices=['before-pretraining', 'after-pretraining'])
    parser.add_argument('--noisy-net-sigma', type=float, default=0.5)

    # Parameters for state/action handling
    parser.add_argument('--frame-stack', type=int, default=None, help='Number of frames stacked (None for disable).')
    parser.add_argument('--frame-skip', type=int, default=None, help='Number of frames skipped (None for disable).')
    parser.add_argument('--camera-atomic-actions', type=int, default=10)
    parser.add_argument('--max-range-of-camera', type=float, default=10.)
    parser.add_argument('--use-full-observation', action='store_true', default=False)
    args = parser.parse_args()

    assert args.expert_demo_path is not None,"DQfD needs collected \
                        expert demonstrations"

    import logging

    if args.logging_filename is not None:
        logging.basicConfig(filename=args.logging_filename, filemode='w',
                            level=args.logging_level)
    else:
        logging.basicConfig(level=args.logging_level)

    logger = logging.getLogger(__name__)

    train_seed = args.seed
    test_seed = 2 ** 31 - 1 - args.seed

    chainerrl.misc.set_random_seed(args.seed, gpus=(args.gpu,))

    args.outdir = experiments.prepare_output_dir(args, args.outdir)
    logger.info('Output files are saved in {}'.format(args.outdir))


    if args.env == 'MineRLTreechop-v0':
        branch_sizes = [9, 16, args.camera_atomic_actions, args.camera_atomic_actions]
    elif args.env in ['MineRLNavigate-v0', 'MineRLNavigateDense-v0',
                    'MineRLNavigateExtreme-v0', 'MineRLNavigateExtremeDense-v0']:
        branch_sizes = [9, 16, args.camera_atomic_actions, args.camera_atomic_actions, 2]
    elif args.env in ['MineRLObtainIronPickaxe-v0', 'MineRLObtainIronPickaxeDense-v0',
                      'MineRLObtainDiamond-v0', 'MineRLObtainDiamondDense-v0']:
        branch_sizes = [9, 16, args.camera_atomic_actions, args.camera_atomic_actions, 32]
    else:
        raise Exception("Unknown environment")

    def make_env(env, test):
        # wrap env: observation...
        # NOTE: wrapping order matters!
        if args.use_full_observation:
            env = FullObservationSpaceWrapper(env)
        elif args.env.startswith('MineRLNavigate'):
            env = PoVWithCompassAngleWrapper(env)
        else:
            env = ObtainPoVWrapper(env)
        if test and args.monitor:
            env = gym.wrappers.Monitor(
                env, os.path.join(args.outdir, 'monitor'),
                mode='evaluation' if test else 'training', video_callable=lambda episode_id: True)
        if args.frame_skip is not None:
            env = FrameSkip(env, skip=args.frame_skip)

        # convert hwc -> chw as Chainer requires
        env = MoveAxisWrapper(env, source=-1, destination=0,
                              use_tuple=args.use_full_observation)
        #env = ScaledFloatFrame(env)
        if args.frame_stack is not None:
            env = FrameStack(env, args.frame_stack, channel_order='chw',
                             use_tuple=args.use_full_observation)

        # wrap env: action...
        env = BranchedActionWrapper(env, branch_sizes, args.camera_atomic_actions, args.max_range_of_camera)

        if test:
            env = BranchedRandomizedAction(env, branch_sizes, args.eval_epsilon)

        env_seed = test_seed if test else train_seed
        env.seed(int(env_seed))
        return env

    core_env = gym.make(args.env)
    env = make_env(core_env, test=False)
    eval_env = make_env(core_env, test=True)

    # Q function
    if args.env.startswith('MineRLNavigate'):
        if args.use_full_observation:
            base_channels = 3  # RGB
        else:
            base_channels = 4  # RGB + compass
    elif args.env.startswith('MineRLObtain'):
        base_channels = 3  # RGB
    else:
        base_channels = 3  # RGB

    if args.frame_stack is None:
        n_input_channels = base_channels
    else:
        n_input_channels = base_channels * args.frame_stack

    q_func = CNNBranchingQFunction(branch_sizes,
                          n_input_channels=n_input_channels,
                          gradient_rescaling=args.gradient_rescaling,
                          use_tuple=args.use_full_observation)

    def phi(x):
        # observation -> NN input
        if args.use_full_observation:
            pov = np.asarray(x[0], dtype=np.float32)
            others = np.asarray(x[1], dtype=np.float32)
            return (pov / 255, others)
        else:
            return np.asarray(x, dtype=np.float32) / 255

    explorer = explorers.LinearDecayEpsilonGreedy(
        1.0, args.final_epsilon,
        args.final_exploration_frames,
        lambda: np.array([np.random.randint(n) for n in branch_sizes]))

    # Draw the computational graph and save it in the output directory.
    if args.use_full_observation:
        sample_obs = tuple([x[None] for x in env.observation_space.sample()])
    else:
        sample_obs = env.observation_space.sample()[None]

    chainerrl.misc.draw_computational_graph(
        [q_func(phi(sample_obs))], os.path.join(args.outdir, 'model'))

    if args.optimizer == 'rmsprop':
        opt = chainer.optimizers.RMSpropGraves(args.lr, alpha=0.95, momentum=0.0, eps=1e-2)
    elif args.optimizer == 'adam':
        opt = chainer.optimizers.Adam(args.lr)

    if args.use_noisy_net is None:
        opt.setup(q_func)

    if args.gradient_rescaling:
        opt.add_hook(ScaleGradHook(1 / (1 + len(q_func.branch_sizes))))
    if args.gradient_clipping:
        opt.add_hook(chainer.optimizer_hooks.GradientClipping(10.0))

    # calculate corresponding `steps` and `eval_interval` according to frameskip
    maximum_frames = 8640000  # = 1440 episodes if we count an episode as 6000 frames.
    if args.frame_skip is None:
        steps = maximum_frames
        eval_interval = 6000 * 100  # (approx.) every 100 episode (counts "1 episode = 6000 steps")
    else:
        steps = maximum_frames // args.frame_skip
        eval_interval = 6000 * 100 // args.frame_skip  # (approx.) every 100 episode (counts "1 episode = 6000 steps")

    # Anneal beta from beta0 to 1 throughout training
    betasteps = steps / args.update_interval
    replay_buffer = PrioritizedDemoReplayBuffer(
        args.replay_buffer_size, alpha=0.4,
        beta0=0.6, betasteps=betasteps,
        error_max=args.error_max,
        num_steps=args.num_step_return)

    # Fill the demo buffer with expert transitions
    if not args.demo:
        chosen_dirs = choose_top_experts(args.expert_demo_path, args.n_experts,
                                         logger=logger)

        fill_buffer(args.env, chosen_dirs, replay_buffer, args.frame_skip,
                    args.frame_stack, args.camera_atomic_actions,
                    args.max_range_of_camera, args.use_full_observation,
                    logger=logger)

        logger.info("Demo buffer loaded with {} transitions".format(
            len(replay_buffer)))

    def reward_transform(x):
        return np.sign(x) * np.log(1 + np.abs(x))

    if args.use_noisy_net is not None and args.use_noisy_net == 'before-pretraining':
        chainerrl.links.to_factorized_noisy(q_func, sigma_scale=args.noisy_net_sigma)
        explorer = explorers.Greedy()

        opt.setup(q_func)

    agent = DQfD(q_func, opt, replay_buffer,
                 gamma=0.99,
                 explorer=explorer,
                 n_pretrain_steps=args.n_pretrain_steps,
                 demo_supervised_margin=args.demo_supervised_margin,
                 bonus_priority_agent=args.bonus_priority_agent,
                 bonus_priority_demo=args.bonus_priority_demo,
                 loss_coeff_nstep=args.loss_coeff_nstep,
                 loss_coeff_supervised=args.loss_coeff_supervised,
                 loss_coeff_l2=args.loss_coeff_l2,
                 gpu=args.gpu,
                 replay_start_size=args.replay_start_size,
                 target_update_interval=args.target_update_interval,
                 clip_delta=args.clip_delta,
                 update_interval=args.update_interval,
                 batch_accumulator=args.batch_accumulator,
                 phi=phi, reward_transform=reward_transform,
                 minibatch_size=args.minibatch_size)

    if args.use_noisy_net is not None and args.use_noisy_net == 'after-pretraining':
        chainerrl.links.to_factorized_noisy(q_func, sigma_scale=args.noisy_net_sigma)
        explorer = explorers.Greedy()

        if args.optimizer == 'rmsprop':
            opt = chainer.optimizers.RMSpropGraves(args.lr, alpha=0.95, momentum=0.0, eps=1e-2)
        elif args.optimizer == 'adam':
            opt = chainer.optimizers.Adam(args.lr)
        opt.setup(q_func)
        opt.add_hook(
            chainer.optimizer_hooks.WeightDecay(args.loss_coeff_l2))
        agent.optimizer = opt

        agent.target_model = None
        agent.sync_target_network()

    if args.load:
        agent.load(args.load)

    if args.demo:
        eval_stats = experiments.eval_performance(
            env=eval_env, agent=agent, n_steps=None, n_episodes=args.eval_n_runs)
        logger.info('n_runs: {} mean: {} median: {} stdev: {}'.format(
            args.eval_n_runs, eval_stats['mean'], eval_stats['median'], eval_stats['stdev']))
    else:
        agent.pretrain()

        evaluator = Evaluator(agent=agent,
                              n_steps=None,
                              n_episodes=args.eval_n_runs,
                              eval_interval=eval_interval,
                              outdir=args.outdir,
                              max_episode_len=None,
                              env=eval_env,
                              step_offset=0,
                              save_best_so_far_agent=True,
                              logger=logger)

        # Evaluate the agent BEFORE training begins
        evaluator.evaluate_and_update_max_score(t=0, episodes=0)

        experiments.train_agent(agent=agent,
                                env=env,
                                steps=steps,
                                outdir=args.outdir,
                                max_episode_len=None,
                                step_offset=0,
                                evaluator=evaluator,
                                successful_score=None,
                                step_hooks=[])

    env.close()


if __name__ == "__main__":
    main()
