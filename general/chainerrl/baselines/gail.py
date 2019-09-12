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

import chainerrl
from chainerrl import experiments
from chainerrl.wrappers.atari_wrappers import (
    FrameStack, ScaledFloatFrame, LazyFrames)
from agents.behavioral_cloning import BehavioralCloning
from agents.gail import Discriminator, GAIL
from expert_dataset import ExpertDataset
from observation_wrappers import (
    generate_pov_converter, generate_pov_with_compass_converter,
    generate_unified_observation_converter)
from action_wrappers import (
    generate_discrete_converter, generate_continuous_converter,
    generate_multi_dimensional_softmax_converter)
from distribution import MultiDimensionalSoftmaxDistribution
from utils import ordinal_logit_function

from env_wrappers import (
    SerialDiscreteActionWrapper, NormalizedContinuousActionWrapper,
    UnifiedObservationWrapper, GrayScaleWrapper, PoVWithCompassAngleWrapper,
    ObtainPoVWrapper, FrameSkip, MoveAxisWrapper,
    MultiDimensionalSoftmaxActionWrapper)

logger = getLogger(__name__)


class ActorTRPONetForDiscrete(chainer.Chain):
    def __init__(self, n_actions, n_input_channels=4, activation=F.relu,
                 bias=0.1, hiddens=None, use_bn=False):
        self.n_input_channels = n_input_channels
        self.activation = activation
        self.hiddens = [512] if hiddens is None else hiddens
        self.use_bn = use_bn

        super(ActorTRPONetForDiscrete, self).__init__()
        with self.init_scope():
            self.conv_layers = chainer.ChainList(
                L.Convolution2D(n_input_channels, 32, 8, stride=4,
                                initial_bias=bias),
                L.Convolution2D(32, 64, 4, stride=2, initial_bias=bias),
                L.Convolution2D(64, 64, 3, stride=1, initial_bias=bias))
            if self.use_bn:
                self.bn_layers = chainer.ChainList(
                    L.BatchNormalization(32),
                    L.BatchNormalization(64),
                    L.BatchNormalization(64))
                self.a_stream = chainerrl.links.mlp_bn.MLPBN(None, n_actions, self.hiddens,
                                                             normalize_input=False)
            else:
                self.a_stream = chainerrl.links.mlp.MLP(None, n_actions, self.hiddens)

    def __call__(self, s):
        h = s
        if self.use_bn:
            for l, b in zip(self.conv_layers, self.bn_layers):
                h = self.activation(b(l(h)))
        else:
            for l in self.conv_layers:
                h = self.activation(l(h))
        out = self.a_stream(h)
        return chainerrl.distribution.SoftmaxDistribution(out)


class ActorTRPONetForContinuous(chainer.Chain):
    def __init__(self, n_actions, n_input_channels=4, activation=F.relu,
                 bias=0.1, var_param_init=0,  # var_func=F.softplus,
                 hiddens=None, use_bn=False):
        self.n_input_channels = n_input_channels
        self.activation = activation
        self.hiddens = [512] if hiddens is None else hiddens
        # self.var_func = var_func
        self.use_bn = use_bn

        super(ActorTRPONetForContinuous, self).__init__()
        with self.init_scope():
            self.conv_layers = chainer.ChainList(
                L.Convolution2D(n_input_channels, 32, 8, stride=4,
                                initial_bias=bias),
                L.Convolution2D(32, 64, 4, stride=2, initial_bias=bias),
                L.Convolution2D(64, 64, 3, stride=1, initial_bias=bias))
            if self.use_bn:
                self.bn_layers = chainer.ChainList(
                    L.BatchNormalization(32),
                    L.BatchNormalization(64),
                    L.BatchNormalization(64))
                self.a_stream = chainerrl.links.mlp_bn.MLPBN(None, n_actions, self.hiddens,
                                                             normalize_input=False)
            else:
                self.a_stream = chainerrl.links.mlp.MLP(None, n_actions, self.hiddens)
            self.var_param = chainer.Parameter(initializer=var_param_init,
                                               shape=(1,))
            # self.var_param = chainer.Parameter(
            #     initializer=var_param_init, shape=(n_actions,))  # independent

    def __call__(self, s):
        h = s
        if self.use_bn:
            for l, b in zip(self.conv_layers, self.bn_layers):
                h = self.activation(b(l(h)))
        else:
            for l in self.conv_layers:
                h = self.activation(l(h))
        mean = F.tanh(self.a_stream(h))
        var = F.broadcast_to(self.var_param, mean.shape)
        # var = F.broadcast_to(self.var_func(self.var_param), mean.shape)
        return chainerrl.distribution.GaussianDistribution(mean, var)


class ActorTRPONetForMultiDimensionalSoftmax(chainer.Chain):
    def __init__(self, action_space, n_input_channels=4, activation=F.relu,
                 bias=0.1, var_param_init=0, hiddens=None, use_bn=False,
                 use_ordinal_logit=False):
        n_actions = action_space.high + 1
        self.n_input_channels = n_input_channels
        self.activation = activation
        self.hiddens = [512] if hiddens is None else hiddens
        self.use_bn = use_bn
        self.use_ordinal_logit = use_ordinal_logit

        super(ActorTRPONetForMultiDimensionalSoftmax, self).__init__()
        with self.init_scope():
            self.conv_layers = chainer.ChainList(
                L.Convolution2D(n_input_channels, 32, 8, stride=4,
                                initial_bias=bias),
                L.Convolution2D(32, 64, 4, stride=2, initial_bias=bias),
                L.Convolution2D(64, 64, 3, stride=1, initial_bias=bias))
            if self.use_bn:
                self.bn_layers = chainer.ChainList(
                    L.BatchNormalization(32),
                    L.BatchNormalization(64),
                    L.BatchNormalization(64))
                self.hidden_layers = chainer.ChainList(
                    *[chainerrl.links.mlp_bn.LinearBN(None, hidden) for hidden in self.hiddens])
                self.action_layers = chainer.ChainList(
                    *[L.Linear(None, n) for n in n_actions])
            else:
                self.hidden_layers = chainer.ChainList(
                    *[L.Linear(None, hidden) for hidden in self.hiddens])
                self.action_layers = chainer.ChainList(
                    *[L.Linear(None, n) for n in n_actions])

    def __call__(self, s):
        h = s
        if self.use_bn:
            for l, b in zip(self.conv_layers, self.bn_layers):
                h = self.activation(b(l(h)))
        else:
            for l in self.conv_layers:
                h = self.activation(l(h))
        for l in self.hidden_layers:
            h = self.activation(l(h))
        out = [layer(h) for layer in self.action_layers]
        if self.use_ordinal_logit:
            out = [ordinal_logit_function(v) for v in out]
        return MultiDimensionalSoftmaxDistribution(out)


class SeparatedActorTRPONetForMultiDimensionalSoftmax(chainer.Chain):
    def __init__(self, action_space, n_input_channels=4, activation=F.relu,
                 bias=0.1, var_param_init=0, hiddens=None, use_bn=False,
                 use_ordinal_logit=False):
        self.n_actions = action_space.high + 1
        self.n_input_channels = n_input_channels
        self.activation = activation
        self.hiddens = [512] if hiddens is None else hiddens
        self.use_bn = use_bn
        self.use_ordinal_logit = use_ordinal_logit

        super(SeparatedActorTRPONetForMultiDimensionalSoftmax, self).__init__()
        with self.init_scope():
            self.conv_layers = chainer.ChainList(
                *[chainer.ChainList(
                    L.Convolution2D(n_input_channels, 32, 8, stride=4,
                                    initial_bias=bias),
                    L.Convolution2D(32, 64, 4, stride=2, initial_bias=bias),
                    L.Convolution2D(64, 64, 3, stride=1, initial_bias=bias))
                for n in self.n_actions])
            if self.use_bn:
                self.bn_layers = [chainer.ChainList(
                    L.BatchNormalization(32),
                    L.BatchNormalization(64),
                    L.BatchNormalization(64))
                    for n in self.n_actions]
                self.hidden_layers = [chainer.ChainList(
                    *[chainerrl.links.mlp_bn.LinearBN(None, hidden) for hidden in self.hiddens])
                    for n in self.n_actions]
                self.action_layers = chainer.ChainList(
                    *[L.Linear(None, n) for n in self.n_actions])
            else:
                self.hidden_layers = chainer.ChainList(
                    *[chainer.ChainList(
                        *[L.Linear(None, hidden) for hidden in self.hiddens])
                    for n in self.n_actions])
                self.action_layers = chainer.ChainList(
                    *[L.Linear(None, n) for n in self.n_actions])

    def __call__(self, s):
        out = []
        for i in range(len(self.n_actions)):
            h = s
            if self.use_bn:
                for l, b in zip(self.conv_layers[i], self.bn_layers[i]):
                    h = self.activation(b(l(h)))
            else:
                for l in self.conv_layers[i]:
                    h = self.activation(l(h))
            for l in self.hidden_layers[i]:
                h = self.activation(l(h))
            out.append(self.action_layers[i](h))
        if self.use_ordinal_logit:
            out = [ordinal_logit_function(v) for v in out]
        return MultiDimensionalSoftmaxDistribution(out)


class ActorVFunc(chainer.Chain):
    def __init__(self, n_input_channels=4, activation=F.relu,
                 bias=0.1, hiddens=None, use_bn=False):
        self.n_input_channels = n_input_channels
        self.activation = activation
        self.hiddens = [512] if hiddens is None else hiddens
        self.use_bn = use_bn

        super(ActorVFunc, self).__init__()
        with self.init_scope():
            self.conv_layers = chainer.ChainList(
                L.Convolution2D(n_input_channels, 32, 8, stride=4,
                                initial_bias=bias),
                L.Convolution2D(32, 64, 4, stride=2, initial_bias=bias),
                L.Convolution2D(64, 64, 3, stride=1, initial_bias=bias))
            if self.use_bn:
                self.bn_layers = chainer.ChainList(
                    L.BatchNormalization(32),
                    L.BatchNormalization(64),
                    L.BatchNormalization(64))
                self.a_stream = chainerrl.links.mlp_bn.MLPBN(None, 1, self.hiddens,
                                                             normalize_input=False)
            else:
                self.a_stream = chainerrl.links.mlp.MLP(None, 1, self.hiddens)

    def __call__(self, s):
        h = s
        if self.use_bn:
            for l, b in zip(self.conv_layers, self.bn_layers):
                h = self.activation(b(l(h)))
        else:
            for l in self.conv_layers:
                h = self.activation(l(h))
        return self.a_stream(h)


class ActorPPONet(chainer.Chain, chainerrl.agents.a3c.A3CModel):
    def __init__(self, policy_model, vf_model):
        super(ActorPPONet, self).__init__()
        with self.init_scope():
            self.pi = policy_model
            self.v = vf_model

    def pi_and_v(self, state):
        return self.pi(state), self.v(state)


class DiscNet(chainer.Chain):
    """Network for discriminator
    Input: images, action
    Output: np.array(shape=(1,))
    """
    def __init__(self, n_input_channels=4, activation=F.relu,
                 bias=0.1, use_dropout=False, use_bn=False,
                 action_wrapper=None, hiddens=None):
        self.n_input_channels = n_input_channels
        self.activation = activation
        if use_dropout:
            self.activation = lambda x: F.dropout(activation(x))
        self.use_bn = use_bn
        self.hiddens = [512, 512] if hiddens is None else hiddens
        assert action_wrapper in ['discrete', 'continuous',
                                  'multi-dimensional-softmax']
        self.action_wrapper = action_wrapper

        super(DiscNet, self).__init__()
        with self.init_scope():
            self.conv_layers = chainer.ChainList(
                L.Convolution2D(n_input_channels, 32, 8, stride=4,
                                initial_bias=bias),
                L.Convolution2D(32, 64, 4, stride=2, initial_bias=bias),
                L.Convolution2D(64, 64, 3, stride=1, initial_bias=bias))
            if self.use_bn:
                self.bn_layers = chainer.ChainList(
                    L.BatchNormalization(32),
                    L.BatchNormalization(64),
                    L.BatchNormalization(64))
                self.a_stream = chainerrl.links.mlp_bn.MLPBN(
                    None, 1, self.hiddens, nonlinearity=self.activation,
                    normalize_input=False)
            else:
                self.a_stream = chainerrl.links.mlp.MLP(
                    None, 1, self.hiddens, nonlinearity=self.activation)

    def __call__(self, s, a):
        num_data = s.shape[0]
        h = s
        if self.use_bn:
            for l, b in zip(self.conv_layers, self.bn_layers):
                h = self.activation(b(l(h)))
        else:
            for l in self.conv_layers:
                h = self.activation(l(h))
        # flatten
        h = F.reshape(h, (num_data, -1))
        if self.action_wrapper == 'discrete':
            a = a.reshape(-1, 1)
        return self.a_stream(F.concat((h, a), axis=1))


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
            env, disable_pitch=(not allow_pitch),
            max_camera_range=max_camera_range)
    elif action_wrapper == 'multi-dimensional-softmax':
        return MultiDimensionalSoftmaxActionWrapper(
            env,
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
    parser.add_argument('--frame-stack', type=int, default=None, help='Number of frames stacked (None for disable).')
    parser.add_argument('--frame-skip', type=int, default=None, help='Number of frames skipped (None for disable).')
    parser.add_argument('--gamma', type=float, default=0.995, help='Discount rate.')
    parser.add_argument('--always-keys', type=str, default=None, nargs='*',
                        help='List of action keys, which should be always pressed throughout interaction with environment.')
    parser.add_argument('--reverse-keys', type=str, default=None, nargs='*',
                        help='List of action keys, which should be always pressed but can be turn off via action.')
    parser.add_argument('--exclude-keys', type=str, default=None, nargs='*',
                        help='List of action keys, which should be ignored for discretizing action space.')
    parser.add_argument('--exclude-noop', action='store_true', default=False, help='The "noop" will be excluded from discrete action list.')
    parser.add_argument('--action-wrapper', type=str, default='discrete',
                        choices=['discrete', 'continuous', 'multi-dimensional-softmax'])
    parser.add_argument('--use-dropout', action='store_true')
    parser.add_argument('--use-ordinal-logit', action='store_true')
    parser.add_argument('--use-noisy-label', action='store_true',
                        help='Add noise on loss of discriminator')
    parser.add_argument('--use-batch-normalization', action='store_true')
    parser.add_argument('--max-camera-range', type=float, default=10.,
                        help='Maximum value of camera angle change in one frame')
    parser.add_argument('--num-camera-discretize', type=int, default=7,
                        help='Number of actions to discretize pitch/yaw respectively')
    parser.add_argument('--activation-function', type=str, default='tanh',
                        choices=['sigmoid', 'tanh', 'relu', 'leaky-relu'])
    parser.add_argument('--prioritized-elements', type=str, nargs='+', default=None,
                        help='define priority of each element on discrete setting')
    parser.add_argument('--allow-pitch', action='store_true', default=False,
                        help='Always set camera pitch as 0 in agent action.')

    parser.add_argument('--policy', type=str, default='trpo', choices=['trpo', 'ppo'])
    parser.add_argument('--discriminator-lr', type=float, default=3e-4)
    parser.add_argument('--policy-lr', type=float, default=3e-4)
    parser.add_argument('--policy-update-interval', type=int, default=1024,
                        help='Interval steps of TRPO iterations.')
    parser.add_argument('--policy-minibatch-size', type=int, default=128)
    parser.add_argument('--discriminator-update-interval', type=int,
                        default=3072,
                        help='Interval steps of Discriminator iterations.')
    parser.add_argument('--discriminator-minibatch-size', type=int, default=3072)
    parser.add_argument('--use-hook', action='store_true')
    parser.add_argument('--policy-entropy-coef', type=float, default=0)
    parser.add_argument('--initial-var-param', type=float, default=0.5)
    parser.add_argument('--discriminator-entropy-coef', type=float,
                        default=1e-3)
    parser.add_argument('--original-reward-weight', type=float, default=0.0,
                        help='define the weight of original reward with discriminator\'s value.')
    parser.add_argument('--separate-multiple-actions', action='store_true')
    parser.add_argument('--pretrain', action='store_true')

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
    # test_seed = 2 ** 31 - 1 - args.seed

    def wrap_env(env, test):
        # wrap env: observation...
        # NOTE: wrapping order matters!
        if args.gray_scale:
            env = GrayScaleWrapper(env, dict_space_key='pov')
        if args.env.startswith('MineRLObtain'):
            env = UnifiedObservationWrapper(env)
        elif args.env.startswith('MineRLNavigate'):
            env = PoVWithCompassAngleWrapper(env)
        else:
            env = ObtainPoVWrapper(env)
        if test and args.monitor:
            env = gym.wrappers.Monitor(
                env, args.outdir, mode='evaluation' if test else 'training', video_callable=lambda episode_id: True)
        if args.frame_skip is not None:
            env = FrameSkip(env, skip=args.frame_skip)
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

        # env_seed = test_seed if test else train_seed
        # env.seed(int(env_seed))  # TODO: not supported yet
        return env

    core_env = gym.make(args.env)
    core_env.seed(int(args.seed))
    env = wrap_env(core_env, test=False)
    # eval_env = gym.make(args.env)  # Can't create multiple MineRL envs
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

    # ================ Set up Policy Networks ================
    if args.activation_function == 'sigmoid':
        activation_func = F.sigmoid
    elif args.activation_function == 'tanh':
        activation_func = F.tanh
    elif args.activation_function == 'relu':
        activation_func = F.relu
    elif args.activation_function == 'leaky-relu':
        activation_func = F.leaky_relu

    n_input_channels = env.observation_space.shape[0]

    # Normalize observations based on their empirical mean and variance
    obs_normalizer = chainerrl.links.EmpiricalNormalization(
        (n_input_channels, 64, 64), clip_threshold=5)

    if args.action_wrapper == 'discrete':
        n_actions = env.action_space.n
        policy = ActorTRPONetForDiscrete(
            n_actions, n_input_channels=n_input_channels,
            activation=activation_func,
            use_bn=args.use_batch_normalization)
    elif args.action_wrapper == 'continuous':
        n_actions = env.action_space.low.shape[0]
        policy = ActorTRPONetForContinuous(
            n_actions, n_input_channels=n_input_channels,
            activation=activation_func,
            var_param_init=args.initial_var_param,
            use_bn=args.use_batch_normalization)
    elif args.action_wrapper == 'multi-dimensional-softmax':
        if args.separate_multiple_actions:
            policy = SeparatedActorTRPONetForMultiDimensionalSoftmax(
                env.action_space, n_input_channels=n_input_channels,
                activation=activation_func,
                var_param_init=args.initial_var_param, use_bn=args.use_batch_normalization,
                use_ordinal_logit=args.use_ordinal_logit)
        else:
            policy = ActorTRPONetForMultiDimensionalSoftmax(
                env.action_space, n_input_channels=n_input_channels,
                activation=activation_func,
                var_param_init=args.initial_var_param, use_bn=args.use_batch_normalization,
                use_ordinal_logit=args.use_ordinal_logit)

    # Use a value function to reduce variance
    vf = ActorVFunc(n_input_channels=n_input_channels,
                    activation=activation_func,
                    use_bn=args.use_batch_normalization)

    # Draw the computational graph and save it in the output directory.
    sample_obs = env.observation_space.sample()
    sample_batch_obs = np.expand_dims(sample_obs, 0)
    chainerrl.misc.draw_computational_graph([policy(sample_batch_obs).logits], os.path.join(args.outdir, 'policy'))
    chainerrl.misc.draw_computational_graph([vf(sample_batch_obs)], os.path.join(args.outdir, 'vf'))

    # ================ Set up Expert dataset ================
    logger.info('Loading expert dataset...')
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
    if args.demo:
        experts = None  # dummy
    else:
        experts = ExpertDataset(
            original_dataset=minerl.data.make(args.env, data_dir=args.expert),
            observation_converter=observation_converter,
            action_converter=action_converter,
            frameskip=args.frame_skip, framestack=args.frame_stack,
            shuffle=False)

    # pretrain
    if args.pretrain:
        bc_opt = chainer.optimizers.Adam(alpha=2.5e-4)
        bc_opt.setup(policy)
        bc_agent = BehavioralCloning(policy, bc_opt,
                                     minibatch_size=1024,
                                     action_wrapper=args.action_wrapper)
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

        num_train_data = experts.size * 7 // 10
        train_obs = all_obs[:num_train_data]
        train_acs = all_action[:num_train_data]
        validate_obs = all_obs[num_train_data:]
        validate_acs = all_action[num_train_data:]
        bc_agent.train(train_obs, train_acs, validate_obs, validate_acs)

    # ================ Set up Policy ================
    if args.policy == 'trpo':
        if args.gpu >= 0:
            chainer.cuda.get_device_from_id(args.gpu).use()
            policy.to_gpu(args.gpu)
            vf.to_gpu(args.gpu)
            obs_normalizer.to_gpu(args.gpu)

        # TRPO's policy is optimized via CG and line search, so it doesn't require
        # a chainer.Optimizer. Only the value function needs it.
        policy_opt = chainer.optimizers.Adam(alpha=args.policy_lr)
        policy_opt.setup(vf)

        """
        # Draw the computational graph and save it in the output directory.
        fake_obs = chainer.Variable(
            policy.xp.zeros_like(
                policy.xp.array(obs_space.low), dtype=np.float32)[None],
            name='observation')
        chainerrl.misc.draw_computational_graph(
            [policy(fake_obs)], os.path.join(args.outdir, 'policy'))
        chainerrl.misc.draw_computational_graph(
            [vf(fake_obs)], os.path.join(args.outdir, 'vf'))
        """

        actor = chainerrl.agents.TRPO(
            policy=policy,
            vf=vf,
            vf_optimizer=policy_opt,
            obs_normalizer=obs_normalizer,
            update_interval=args.policy_update_interval,
            conjugate_gradient_max_iter=10,
            conjugate_gradient_damping=1e-1,
            gamma=args.gamma,
            lambd=0.97,
            vf_epochs=5,
            vf_batch_size=args.policy_minibatch_size,
            entropy_coef=args.policy_entropy_coef,
        )
    elif args.policy == 'ppo':
        ppo_policy = ActorPPONet(policy, vf)
        policy_opt = chainer.optimizers.Adam(alpha=args.policy_lr)
        policy_opt.setup(ppo_policy)

        actor = chainerrl.agents.PPO(
            ppo_policy, policy_opt,
            obs_normalizer=obs_normalizer,
            gpu=args.gpu,
            update_interval=args.policy_update_interval,
            minibatch_size=args.policy_minibatch_size,
            epochs=1,
            gamma=args.gamma,
            clip_eps_vf=None,
            entropy_coef=args.policy_entropy_coef,
            standardize_advantages=False,
        )
    else:
        raise RuntimeError('Unsupported policy name: {}'.format(args.policy))

    # ================ Set up Discriminator ================
    obs_normalizer_disc = chainerrl.links.EmpiricalNormalization(
        (n_input_channels, 64, 64), clip_threshold=5)
    disc_model = DiscNet(n_input_channels=n_input_channels,
                         use_dropout=args.use_dropout,
                         activation=activation_func,
                         action_wrapper=args.action_wrapper,
                         use_bn=args.use_batch_normalization)

    # Draw the computational graph and save it in the output directory.
    sample_obs = env.observation_space.sample()
    sample_batch_obs = np.expand_dims(sample_obs, 0)
    sample_action = env.action_space.sample()
    sample_batch_action = np.expand_dims(sample_action, 0).astype(np.float32)
    chainerrl.misc.draw_computational_graph([disc_model(sample_batch_obs, sample_batch_action)], os.path.join(args.outdir, 'discriminator'))

    disc_opt = chainer.optimizers.Adam(alpha=args.discriminator_lr)
    disc_opt.setup(disc_model)
    discriminator = Discriminator(
        disc_model, disc_opt, obs_normalizer=obs_normalizer_disc,
        update_interval=args.discriminator_update_interval,
        minibatch_size=args.discriminator_minibatch_size,
        entropy_coef=args.discriminator_entropy_coef,
        noisy_label=args.use_noisy_label, gpu=args.gpu)

    # ================ Set up GAIL ================
    agent = GAIL(actor, discriminator, experts, args.original_reward_weight)

    if args.load:
        agent.load(args.load)

    # ================================

    # experiment
    if args.demo:
        eval_stats = experiments.eval_performance(env=eval_env, agent=agent, n_steps=None, n_episodes=args.eval_n_runs)
        logger.info('n_runs: {} mean: {} median: {} stdev {}'.format(
            args.eval_n_runs, eval_stats['mean'], eval_stats['median'], eval_stats['stdev']))
    else:
        step_hooks = []
        if args.use_hook:
            if args.action_wrapper == 'continuous':
                # set variance hook
                xp = agent.discriminator.model.xp
                if args.policy == 'trpo':
                    def variance_setter(env, agent, value):
                        agent.policy.model.var_param.array = xp.reshape(
                            xp.array(value, dtype=xp.float32),
                            (1, ))
                elif args.policy == 'ppo':
                    def variance_setter(env, agent, value):
                        agent.policy.model.pi.var_param.array = xp.reshape(
                            xp.array(value, dtype=xp.float32),
                            (1, ))

                variance_decay_hook = experiments.LinearInterpolationHook(
                    steps, args.initial_var_param, 1e-2,
                    variance_setter)

                step_hooks.append(variance_decay_hook)

            def policy_lr_setter(env, agent, value):
                agent.policy.optimizer.alpha = value

            policy_lr_decay_hook = experiments.LinearInterpolationHook(
                steps, args.policy_lr, 0, policy_lr_setter)

            step_hooks.append(policy_lr_decay_hook)

            def discriminator_lr_setter(env, agent, value):
                agent.discriminator.optimizer.alpha = value

            discriminator_lr_decay_hook = experiments.LinearInterpolationHook(
                steps, args.discriminator_lr, 0, discriminator_lr_setter)

            step_hooks.append(discriminator_lr_decay_hook)

        experiments.train_agent_with_evaluation(
            agent=agent, env=env, steps=steps,
            eval_n_steps=None,
            eval_n_episodes=args.eval_n_runs,
            eval_interval=eval_interval,
            outdir=args.outdir,
            save_best_so_far_agent=True,
            eval_env=eval_env,
            step_hooks=step_hooks,
        )

    env.close()
    eval_env.close()


if __name__ == '__main__':
    main()
