"""original source: https://github.com/chainer/chainerrl/blob/master/examples/atari/train_ppo_ale.py

MIT License

Copyright (c) Preferred Networks, Inc.
"""

import argparse
from logging import getLogger
import os

import minerl  # noqa: register MineRL envs as Gym envs.
import gym
import numpy as np

import chainer

import chainerrl
from chainerrl.wrappers import ContinuingTimeLimit
from chainerrl.wrappers.atari_wrappers import FrameStack, ScaledFloatFrame

import sys
sys.path.append(os.path.abspath(os.path.join(__file__, os.pardir)))
import utils
from q_functions import NatureDQNHead, A3CFF
from env_wrappers import (
    SerialDiscreteActionWrapper, CombineActionWrapper, SerialDiscreteCombineActionWrapper,
    ContinuingTimeLimitMonitor,
    MoveAxisWrapper, FrameSkip, ObtainPoVWrapper, PoVWithCompassAngleWrapper, GrayScaleWrapper)

logger = getLogger(__name__)


def parse_arch(arch, n_actions, n_input_channels):
    if arch == 'nature':
        head = NatureDQNHead(n_input_channels=n_input_channels, n_output_channels=512)
    else:
        raise RuntimeError('Unsupported architecture name: {}'.format(arch))
    return A3CFF(n_actions, head)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='MineRLTreechop-v0',
                        choices=[
                            'MineRLTreechop-v0',
                            'MineRLNavigate-v0', 'MineRLNavigateDense-v0',
                            'MineRLNavigateExtreme-v0', 'MineRLNavigateExtremeDense-v0',
                            'MineRLObtainIronPickaxe-v0', 'MineRLObtainIronPickaxeDense-v0',
                            'MineRLObtainDiamond-v0', 'MineRLObtainDiamondDense-v0',
                            # for debug use
                            'MineRLNavigateDenseFixed-v0',
                            'MineRLObtainTest-v0',
                        ],
                        help='MineRL environment identifier.')
    parser.add_argument('--outdir', type=str, default='results',
                        help='Directory path to save output files. If it does not exist, it will be created.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed [0, 2 ** 31)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use, set to -1 if no GPU.')
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--arch', type=str, default='nature', choices=['nature'],
                        help='Network architecture to use.')
    # In the original paper, agent runs in 8 environments parallely and samples 128 steps per environment.
    # Sample 128 * 8 steps, instead.
    parser.add_argument('--update-interval', type=int, default=128 * 8, help='Frequency (in timesteps) of network updates.')
    parser.add_argument('--eval-n-runs', type=int, default=3)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--logging-level', type=int, default=20, help='Logging level. 10:DEBUG, 20:INFO etc.')
    parser.add_argument('--gray-scale', action='store_true', default=False, help='Convert pov into gray scaled image.')
    parser.add_argument('--monitor', action='store_true', default=False,
                        help='Monitor env. Videos and additional information are saved as output files when evaluation.')
    parser.add_argument('--lr', type=float, default=2.5e-4, help='Learning rate.')
    parser.add_argument('--adam-eps', type=float, default=1e-8, help='Epsilon for Adam.')
    parser.add_argument('--frame-stack', type=int, default=None, help='Number of frames stacked (None for disable).')
    parser.add_argument('--frame-skip', type=int, default=None, help='Number of frames skipped (None for disable).')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount rate.')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs to update model for per PPO iteration.')
    parser.add_argument('--standardize-advantages', action='store_true', default=False, help='Use standardized advantages on updates for PPO')
    parser.add_argument('--disable-action-prior', action='store_true', default=False,
                        help='If specified, action_space shaping based on prior knowledge will be disabled.')
    parser.add_argument('--always-keys', type=str, default=None, nargs='*',
                        help='List of action keys, which should be always pressed throughout interaction with environment.')
    parser.add_argument('--reverse-keys', type=str, default=None, nargs='*',
                        help='List of action keys, which should be always pressed but can be turn off via action.')
    parser.add_argument('--exclude-keys', type=str, default=None, nargs='*',
                        help='List of action keys, which should be ignored for discretizing action space.')
    parser.add_argument('--exclude-noop', action='store_true', default=False, help='The "noop" will be excluded from discrete action list.')
    args = parser.parse_args()

    args.outdir = chainerrl.experiments.prepare_output_dir(args, args.outdir)

    import logging
    log_format = '%(levelname)-8s - %(asctime)s - [%(name)s %(funcName)s %(lineno)d] %(message)s'
    logging.basicConfig(filename=os.path.join(args.outdir, 'log.txt'), format=log_format, level=args.logging_level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(args.logging_level)
    console_handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger('').addHandler(console_handler)  # add hander to the root logger

    logger.info('Output files are saved in {}'.format(args.outdir))

    utils.log_versions()

    try:
        _main(args)
    except:  # noqa
        logger.exception('execution failed.')
        raise


def _main(args):
    logger.info('The first `gym.make(MineRL*)` may take several minutes. Be patient!')

    os.environ['MALMO_MINECRAFT_OUTPUT_LOGDIR'] = args.outdir

    # Set a random seed used in ChainerRL.
    chainerrl.misc.set_random_seed(args.seed)

    # Set different random seeds for train and test envs.
    train_seed = args.seed  # noqa: never used in this script
    test_seed = 2 ** 31 - 1 - args.seed

    def wrap_env(env, test):
        # wrap env: time limit...
        if isinstance(env, gym.wrappers.TimeLimit):
            logger.info('Detected `gym.wrappers.TimeLimit`! Unwrap it and re-wrap our own time limit.')
            env = env.env
            max_episode_steps = env.spec.max_episode_steps
            env = ContinuingTimeLimit(env, max_episode_steps=max_episode_steps)

        # wrap env: observation...
        # NOTE: wrapping order matters!

        if test and args.monitor:
            env = ContinuingTimeLimitMonitor(
                env, os.path.join(args.outdir, 'monitor'),
                mode='evaluation' if test else 'training', video_callable=lambda episode_id: True)
        if args.frame_skip is not None:
            env = FrameSkip(env, skip=args.frame_skip)
        if args.gray_scale:
            env = GrayScaleWrapper(env, dict_space_key='pov')
        if args.env.startswith('MineRLNavigate'):
            env = PoVWithCompassAngleWrapper(env)
        else:
            env = ObtainPoVWrapper(env)
        env = MoveAxisWrapper(env, source=-1, destination=0)  # convert hwc -> chw as Chainer requires.
        env = ScaledFloatFrame(env)
        if args.frame_stack is not None and args.frame_stack > 0:
            env = FrameStack(env, args.frame_stack, channel_order='chw')

        # wrap env: action...
        if not args.disable_action_prior:
            env = SerialDiscreteActionWrapper(
                env,
                always_keys=args.always_keys, reverse_keys=args.reverse_keys, exclude_keys=args.exclude_keys, exclude_noop=args.exclude_noop)
        else:
            env = CombineActionWrapper(env)
            env = SerialDiscreteCombineActionWrapper(env)

        env_seed = test_seed if test else train_seed
        # env.seed(int(env_seed))  # TODO: not supported yet
        return env

    core_env = gym.make(args.env)
    env = wrap_env(core_env, test=False)
    # eval_env = gym.make(args.env)  # Can't create multiple MineRL envs
    # eval_env = wrap_env(eval_env, test=True)
    eval_env = wrap_env(core_env, test=True)

    # model
    n_actions = env.action_space.n
    model = parse_arch(args.arch, n_actions, n_input_channels=env.observation_space.shape[0])

    opt = chainer.optimizers.Adam(alpha=args.lr, eps=args.adam_eps)
    opt.setup(model)
    opt.add_hook(chainer.optimizer.GradientClipping(40))
    if args.weight_decay > 0:
        opt.add_hook(chainerrl.optimizers.nonbias_weight_decay.NonbiasWeightDecay(args.weight_decay))

    # Draw the computational graph and save it in the output directory.
    sample_obs = env.observation_space.sample()
    sample_batch_obs = np.expand_dims(sample_obs, 0)
    chainerrl.misc.draw_computational_graph([model(sample_batch_obs)], os.path.join(args.outdir, 'model'))

    # calculate corresponding `steps` and `eval_interval` according to frameskip
    # = 1333 episodes if we count an episode as 6000 frames,
    # = 1000 episodes if we count an episode as 8000 frames.
    maximum_frames = 8000000
    if args.frame_skip is None:
        steps = maximum_frames
        eval_interval = 6000 * 100  # (approx.) every 100 episode (counts "1 episode = 6000 steps")
    else:
        steps = maximum_frames // args.frame_skip
        eval_interval = 6000 * 100 // args.frame_skip  # (approx.) every 100 episode (counts "1 episode = 6000 steps")

    # build agent
    def phi(x):
        # observation -> NN input
        return np.asarray(x)
    CLIP_EPS = 0.1
    agent = chainerrl.agents.ppo.PPO(
        model, opt, gpu=args.gpu, gamma=args.gamma, phi=phi, update_interval=args.update_interval,
        minibatch_size=32, epochs=args.epochs, clip_eps=CLIP_EPS, standardize_advantages=args.standardize_advantages)
    if args.load:
        agent.load(args.load)

    # experiment
    if args.demo:
        eval_stats = chainerrl.experiments.eval_performance(env=eval_env, agent=agent, n_steps=None, n_episodes=args.eval_n_runs)
        logger.info('n_runs: {} mean: {} median: {} stdev {}'.format(
            args.eval_n_runs, eval_stats['mean'], eval_stats['median'], eval_stats['stdev']))
    else:
        # Linearly decay the learning rate to zero
        def lr_setter(env, agent, value):
            agent.optimizer.alpha = value

        lr_decay_hook = chainerrl.experiments.LinearInterpolationHook(
            steps, args.lr, 0, lr_setter)

        # Linearly decay the clipping parameter to zero
        def clip_eps_setter(env, agent, value):
            agent.clip_eps = max(value, 1e-8)

        clip_eps_decay_hook = chainerrl.experiments.LinearInterpolationHook(
            steps, CLIP_EPS, 0, clip_eps_setter)

        chainerrl.experiments.train_agent_with_evaluation(
            agent=agent, env=env, steps=steps,
            eval_n_steps=None, eval_n_episodes=args.eval_n_runs, eval_interval=eval_interval,
            outdir=args.outdir, eval_env=eval_env,
            step_hooks=[lr_decay_hook, clip_eps_decay_hook],
            save_best_so_far_agent=True,
        )

    env.close()
    eval_env.close()


if __name__ == '__main__':
    main()
