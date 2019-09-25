"""
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
from chainer import functions as F
from chainer import links as L
from chainer import optimizers

import chainerrl
from chainerrl import experiments
from chainerrl.wrappers.atari_wrappers import (
    FrameStack, ScaledFloatFrame, LazyFrames)
from chainerrl.distribution import SoftmaxDistribution, ContinuousDeterministicDistribution
from expert_dataset import ExpertDataset
from observation_wrappers import (
    generate_pov_converter, generate_pov_with_compass_converter,
    generate_unified_observation_converter)
from action_wrappers import (
    generate_discrete_converter, generate_continuous_converter,
    generate_multi_dimensional_softmax_converter)
from agents.behavioral_cloning import BehavioralCloning
from distribution import MultiDimensionalSoftmaxDistribution

from env_wrappers import (
    SerialDiscreteActionWrapper, NormalizedContinuousActionWrapper,
    UnifiedObservationWrapper, GrayScaleWrapper, PoVWithCompassAngleWrapper,
    ObtainPoVWrapper, FrameSkip, MoveAxisWrapper,
    MultiDimensionalSoftmaxActionWrapper)

logger = getLogger(__name__)


class SingleSharedBias(chainer.Chain):
    """Single shared bias used in the Double DQN paper.
    You can add this link after a Linear layer with nobias=True to implement a
    Linear layer with a single shared bias parameter.
    See http://arxiv.org/abs/1509.06461.
    """

    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.bias = chainer.Parameter(0, shape=1)

    def __call__(self, x):
        return x + F.broadcast_to(self.bias, x.shape)


class Net(chainer.Chain):
    """Network of discrete and continuous for behavioral cloning
    Input: images
    Output: np.array(shape=(n_actions,))
    """
    def __init__(self, n_actions, n_input_channels=4, activation=F.relu, bias=0.1, hiddens=None,
                 action_wrapper=None):
        self.n_actions = n_actions
        self.n_input_channels = n_input_channels
        self.activation = activation
        self.hiddens = [512] if hiddens is None else hiddens
        assert action_wrapper in ['discrete', 'continuous']
        self.action_wrapper = action_wrapper

        super().__init__()
        with self.init_scope():
            self.conv_layers = chainer.ChainList(
                L.Convolution2D(n_input_channels, 32, 8, stride=4,
                                initial_bias=bias),
                L.Convolution2D(32, 64, 4, stride=2, initial_bias=bias),
                L.Convolution2D(64, 64, 3, stride=1, initial_bias=bias))
            self.a_stream = chainerrl.links.mlp.MLP(None, n_actions, self.hiddens)

    def get_raw_value(self, x):
        h = np.array(x)
        for l in self.conv_layers:
            h = self.activation(l(h))
        out = self.a_stream(h)
        if self.action_wrapper == 'discrete':
            return out
        else:
            return F.tanh(out)

    def __call__(self, x):
        if self.action_wrapper == 'discrete':
            return SoftmaxDistribution(self.get_raw_value(x),
                                       min_prob=0.0)
        else:
            return ContinuousDeterministicDistribution(self.get_raw_value(x))


class NetForMultiDimensionalSoftmax(chainer.Chain):
    """Network of multi dimensional discrete for behavioral cloning
    Input: images
    Output: a list of np.arrays. Each element represents an action for each action type.
    """
    def __init__(self, action_space, n_input_channels=4,
                 activation=F.relu, bias=0.1, hiddens=None):
        n_actions = action_space.high + 1
        self.n_input_channels = n_input_channels
        self.activation = activation
        self.hiddens = [512] if hiddens is None else hiddens

        super(NetForMultiDimensionalSoftmax, self).__init__()
        with self.init_scope():
            self.conv_layers = chainer.ChainList(
                L.Convolution2D(n_input_channels, 32, 8, stride=4,
                                initial_bias=bias),
                L.Convolution2D(32, 64, 4, stride=2, initial_bias=bias),
                L.Convolution2D(64, 64, 3, stride=1, initial_bias=bias))
            self.hidden_layers = chainer.ChainList(
                *[L.Linear(None, hidden) for hidden in self.hiddens])
            self.action_layers = chainer.ChainList(
                *[L.Linear(None, n) for n in n_actions])

    def get_raw_value(self, x):
        h = x
        for l in self.conv_layers:
            h = self.activation(l(h))
        for l in self.hidden_layers:
            h = self.activation(l(h))
        out = [layer(h) for layer in self.action_layers]
        return out

    def __call__(self, x):
        return MultiDimensionalSoftmaxDistribution(self.get_raw_value(x))


def parse_action_wrapper(action_wrapper, env, always_keys, reverse_keys,
                         exclude_keys, exclude_noop, allow_pitch,
                         num_camera_discretize, max_camera_range):
    if action_wrapper == 'discrete':
        return SerialDiscreteActionWrapper(
            env,
            always_keys=always_keys, reverse_keys=reverse_keys, exclude_keys=exclude_keys, exclude_noop=exclude_noop,
            num_camera_discretize=num_camera_discretize, allow_pitch=allow_pitch,
            max_camera_range=max_camera_range)
    elif action_wrapper == 'continuous':
        return NormalizedContinuousActionWrapper(
            env, allow_pitch=allow_pitch,
            max_camera_range=max_camera_range)
    elif action_wrapper == 'multi-dimensional-softmax':
        return MultiDimensionalSoftmaxActionWrapper(
            env, allow_pitch=allow_pitch,
            num_camera_discretize=num_camera_discretize,
            max_camera_range=max_camera_range)
    else:
        raise RuntimeError('Unsupported action wrapper name: {}'.format(action_wrapper))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='MineRLTreechop-v0',
                        choices=[
                            'MineRLTreechop-v0',
                            'MineRLNavigate-v0', 'MineRLNavigateDense-v0', 'MineRLNavigateExtreme-v0', 'MineRLNavigateExtremeDense-v0',
                            'MineRLObtainIronPickaxe-v0', 'MineRLObtainDiamond-v0',
                            'MineRLObtainIronPickaxeDense-v0', 'MineRLObtainDiamondDense-v0',
                            'MineRLNavigateDenseFixed-v0'  # for debug use
                        ],
                        help='MineRL environment identifier.')
    parser.add_argument('--outdir', type=str, default='results',
                        help='Directory path to save output files. If it does not exist, it will be created.')
    parser.add_argument('--expert', type=str, required=True,
                        help='Path storing expert trajectories.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed [0, 2 ** 31)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use, set to -1 if no GPU.')
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--eval-n-runs', type=int, default=3)
    parser.add_argument('--logging-level', type=int, default=20, help='Logging level. 10:DEBUG, 20:INFO etc.')
    parser.add_argument('--gray-scale', action='store_true', default=False, help='Convert pov into gray scaled image.')
    parser.add_argument('--monitor', action='store_true', default=False,
                        help='Monitor env. Videos and additional information are saved as output files when evaluation.')
    parser.add_argument('--lr', type=float, default=2.5e-4, help='Learning rate.')
    parser.add_argument('--frame-stack', type=int, default=None, help='Number of frames stacked (None for disable).')
    parser.add_argument('--frame-skip', type=int, default=None, help='Number of frames skipped (None for disable).')
    parser.add_argument('--always-keys', type=str, default=None, nargs='*',
                        help='List of action keys, which should be always pressed throughout interaction with environment.')
    parser.add_argument('--reverse-keys', type=str, default=None, nargs='*',
                        help='List of action keys, which should be always pressed but can be turn off via action.')
    parser.add_argument('--exclude-keys', type=str, default=None, nargs='*',
                        help='List of action keys, which should be ignored for discretizing action space.')
    parser.add_argument('--exclude-noop', action='store_true', default=False, help='The "noop" will be excluded from discrete action list.')
    parser.add_argument('--action-wrapper', type=str, default='discrete',
                        choices=['discrete', 'continuous', 'multi-dimensional-softmax'])
    parser.add_argument('--activation-function', type=str, default='tanh',
                        choices=['sigmoid', 'tanh', 'relu', 'leaky-relu'])
    parser.add_argument('--prioritized-elements', type=str, nargs='+', default=None,
                        help='define priority of each element on discrete setting')
    # parser.add_argument('--write-video', type=str, default=None)
    parser.add_argument('--entropy-coef', type=float, default=0)
    parser.add_argument('--allow-pitch', action='store_true', default=False,
                        help='Always set camera pitch as 0 in agent action.')
    parser.add_argument('--max-camera-range', type=float, default=10.,
                        help='Maximum value of camera angle change in one frame')
    parser.add_argument('--num-camera-discretize', type=int, default=7,
                        help='Number of actions to discretize pitch/yaw respectively')
    parser.add_argument('--training-dataset-ratio', type=float, default=0.7,
                        help='ratio of training dataset on behavioral cloning between (0, 1)')
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
        # wrap env: observation...
        # NOTE: wrapping order matters!
        if test and args.monitor:
            env = gym.wrappers.Monitor(
                env, args.outdir, mode='evaluation' if test else 'training', video_callable=lambda episode_id: True)
        if args.gray_scale:
            env = GrayScaleWrapper(env, dict_space_key='pov')
        if args.frame_skip is not None:
            env = FrameSkip(env, skip=args.frame_skip)
        if args.env.startswith('MineRLObtain'):
            env = UnifiedObservationWrapper(env)
        elif args.env.startswith('MineRLNavigate'):
            env = PoVWithCompassAngleWrapper(env)
        else:
            env = ObtainPoVWrapper(env)
        env = MoveAxisWrapper(env, source=-1, destination=0)  # convert hwc -> chw as Chainer requires.
        env = ScaledFloatFrame(env)
        if args.frame_stack is not None:
            env = FrameStack(env, args.frame_stack, channel_order='chw')

        # wrap env: action...
        env = parse_action_wrapper(
            args.action_wrapper,
            env,
            always_keys=args.always_keys, reverse_keys=args.reverse_keys,
            exclude_keys=args.exclude_keys, exclude_noop=args.exclude_noop,
            allow_pitch=args.allow_pitch,
            num_camera_discretize=args.num_camera_discretize,
            max_camera_range=args.max_camera_range)

        env_seed = test_seed if test else train_seed
        env.seed(int(env_seed))  # TODO: not supported yet
        return env

    core_env = gym.make(args.env)
    env = wrap_env(core_env, test=False)
    # eval_env = gym.make(args.env)  # Can't create multiple MineRL envs
    # eval_env = wrap_env(eval_env, test=True)
    eval_env = wrap_env(core_env, test=True)

    # calculate corresponding `steps` and `eval_interval` according to frameskip
    # = 1440 episodes if we count an episode as 6000 frames
    # = 1080 episodes if we count an episode as 8000 frames
    maximum_frames = 8000000
    if args.frame_skip is None:
        steps = maximum_frames
        eval_interval = 6000 * 100  # (approx.) every 100 episode (counts "1 episode = 6000 steps")
    else:
        steps = maximum_frames // args.frame_skip
        eval_interval = 6000 * 100 // args.frame_skip  # (approx.) every 100 episode (counts "1 episode = 6000 steps")

    if args.activation_function == 'sigmoid':
        activation_func = F.sigmoid
    elif args.activation_function == 'tanh':
        activation_func = F.tanh
    elif args.activation_function == 'relu':
        activation_func = F.relu
    elif args.activation_function == 'leaky-relu':
        activation_func = F.leaky_relu

    # Policy network
    n_input_channels = env.observation_space.shape[0]
    if args.action_wrapper == 'multi-dimensional-softmax':
        policy = NetForMultiDimensionalSoftmax(
            env.action_space, n_input_channels=n_input_channels,
            activation=activation_func)
    else:
        if args.action_wrapper == 'discrete':
            n_actions = env.action_space.n
        else:
            n_actions = env.action_space.low.shape[0]
        policy = Net(
            n_actions, n_input_channels=n_input_channels,
            activation=activation_func,
            action_wrapper=args.action_wrapper)

    # Draw the computational graph and save it in the output directory.
    sample_obs = env.observation_space.sample()
    sample_batch_obs = np.expand_dims(sample_obs, 0)
    chainerrl.misc.draw_computational_graph([policy(sample_batch_obs).params], os.path.join(args.outdir, 'model'))

    opt = optimizers.Adam(alpha=args.lr)

    opt.setup(policy)

    # ================ Set up Policy ================
    agent = BehavioralCloning(policy, opt,
                              minibatch_size=1024,
                              entropy_coef=args.entropy_coef,
                              action_wrapper=args.action_wrapper)

    if args.load:
        agent.load(args.load)

    if not args.demo:
        # ================ Set up ExpertDataset ================
        if args.env.startswith('MineRLObtain'):
            observation_converter = generate_unified_observation_converter(
                grayscale=args.gray_scale)
        elif args.env.startswith('MineRLNavigate'):
            observation_converter = generate_pov_with_compass_converter(
                grayscale=args.gray_scale)
        else:
            observation_converter = generate_pov_converter(
                grayscale=args.gray_scale)
        if args.action_wrapper == 'discrete':
            action_converter = generate_discrete_converter(
                args.env, args.prioritized_elements,
                args.always_keys, args.reverse_keys,
                args.exclude_keys, args.exclude_noop,
                args.allow_pitch, args.max_camera_range,
                args.num_camera_discretize)
        elif args.action_wrapper == 'continuous':
            action_converter = generate_continuous_converter(
                args.env, args.allow_pitch, args.max_camera_range)
        elif args.action_wrapper == 'multi-dimensional-softmax':
            action_converter = generate_multi_dimensional_softmax_converter(
                args.allow_pitch, args.max_camera_range, args.num_camera_discretize)

        experts = ExpertDataset(
            original_dataset=minerl.data.make(args.env, data_dir=args.expert),
            observation_converter=observation_converter,
            action_converter=action_converter,
            frameskip=args.frame_skip, framestack=args.frame_stack,
            shuffle=False)

        # separate train data and validation data
        all_obs = []
        all_action = []
        for i in range(experts.size):
            obs, action, _, _, _ = experts.sample()
            all_obs.append(obs)
            all_action.append(action)
        if args.frame_stack > 1:
            all_obs = np.array(all_obs, dtype=LazyFrames)
        else:
            all_obs = np.array(all_obs)
        all_action = np.array(all_action)
        if args.action_wrapper == 'discrete':
            logger.debug('Action histogram:',
                         np.histogram(all_action, bins=n_actions)[0])

        num_train_data = int(experts.size * args.training_dataset_ratio)
        train_obs = all_obs[:num_train_data]
        train_acs = all_action[:num_train_data]
        validate_obs = all_obs[num_train_data:]
        validate_acs = all_action[num_train_data:]

        # Behavioral Cloning
        agent.train(train_obs, train_acs, validate_obs, validate_acs)

    # experiment
    if args.demo:
        eval_stats = experiments.eval_performance(env=eval_env, agent=agent, n_steps=None, n_episodes=args.eval_n_runs)
        logger.info('n_runs: {} mean: {} median: {} stdev {}'.format(
            args.eval_n_runs, eval_stats['mean'], eval_stats['median'], eval_stats['stdev']))
    else:
        experiments.train_agent_with_evaluation(
            agent=agent, env=env, steps=steps,
            eval_n_steps=None,
            eval_n_episodes=args.eval_n_runs,
            eval_interval=eval_interval,
            outdir=args.outdir,
            save_best_so_far_agent=True,
            eval_env=eval_env,
        )

    env.close()
    eval_env.close()


if __name__ == '__main__':
    main()
