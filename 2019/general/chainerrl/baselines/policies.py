import numpy as np

import chainer
from chainer import functions as F
from chainer import links as L

import chainerrl
from chainerrl.distribution import SoftmaxDistribution, ContinuousDeterministicDistribution
from distribution import MultiDimensionalSoftmaxDistribution


class BCNet(chainer.Chain):
    """Network of discrete and continuous for behavioral cloning
    Input: images
    Output: np.array(shape=(n_actions,))
    """
    def __init__(self, n_actions, n_input_channels=4, activation=F.relu, bias=0.1, hiddens=None,
                 action_wrapper=None, use_bn=False, modify_compassangle=False):
        self.n_actions = n_actions
        self.n_input_channels = n_input_channels
        self.activation = activation
        self.hiddens = [512] if hiddens is None else hiddens
        self.use_bn = use_bn
        assert action_wrapper in ['discrete', 'continuous']
        self.action_wrapper = action_wrapper
        self.modify_compassangle = modify_compassangle

        super().__init__()
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

    def get_raw_value(self, x):
        h = np.array(x)
        if self.modify_compassangle:
            h = np.concatenate((
                h[:-1],
                np.clip(h[-1:] * 10, -1, 1)))
        if self.use_bn:
            for l, b in zip(self.conv_layers, self.bn_layers):
                h = self.activation(b(l(h)))
        else:
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


class BCNetForMultiDimensionalSoftmax(chainer.Chain):
    """Network of multi dimensional discrete for behavioral cloning
    Input: images
    Output: a list of np.arrays. Each element represents an action for each action type.
    """
    def __init__(self, action_space, n_input_channels=4,
                 activation=F.relu, bias=0.1, hiddens=None,
                 use_bn=False, use_ordinal_logit=False):
        n_actions = action_space.high + 1
        self.n_input_channels = n_input_channels
        self.activation = activation
        self.hiddens = [512] if hiddens is None else hiddens
        self.use_bn = use_bn
        self.use_ordinal_logit = use_ordinal_logit

        super(BCNetForMultiDimensionalSoftmax, self).__init__()
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

    def get_raw_value(self, x):
        h = x
        if self.use_bn:
            for l, b in zip(self.conv_layers, self.bn_layers):
                h = self.activation(b(l(h)))
        else:
            for l in self.conv_layers:
                h = self.activation(l(h))
        for l in self.hidden_layers:
            h = self.activation(l(h))
        out = [layer(h) for layer in self.action_layers]
        return out

    def __call__(self, x):
        return MultiDimensionalSoftmaxDistribution(self.get_raw_value(x))


class ActorTRPONetForDiscrete(chainer.Chain):
    def __init__(self, n_actions, n_input_channels=4, activation=F.relu,
                 bias=0.1, hiddens=None):
        self.n_input_channels = n_input_channels
        self.activation = activation
        self.hiddens = [512] if hiddens is None else hiddens

        super(ActorTRPONetForDiscrete, self).__init__()
        with self.init_scope():
            self.conv_layers = chainer.ChainList(
                L.Convolution2D(n_input_channels, 32, 8, stride=4,
                                initial_bias=bias),
                L.Convolution2D(32, 64, 4, stride=2, initial_bias=bias),
                L.Convolution2D(64, 64, 3, stride=1, initial_bias=bias))
            self.a_stream = chainerrl.links.mlp.MLP(None, n_actions, self.hiddens)

    def __call__(self, s):
        h = s
        for l in self.conv_layers:
            h = self.activation(l(h))
        out = self.a_stream(h)
        return chainerrl.distribution.SoftmaxDistribution(out)


class ActorTRPONetForContinuous(chainer.Chain):
    def __init__(self, n_actions, n_input_channels=4, activation=F.relu,
                 bias=0.1, var_param_init=0,  # var_func=F.softplus,
                 hiddens=None):
        self.n_input_channels = n_input_channels
        self.activation = activation
        self.hiddens = [512] if hiddens is None else hiddens
        # self.var_func = var_func

        super(ActorTRPONetForContinuous, self).__init__()
        with self.init_scope():
            self.conv_layers = chainer.ChainList(
                L.Convolution2D(n_input_channels, 32, 8, stride=4,
                                initial_bias=bias),
                L.Convolution2D(32, 64, 4, stride=2, initial_bias=bias),
                L.Convolution2D(64, 64, 3, stride=1, initial_bias=bias))
            self.a_stream = chainerrl.links.mlp.MLP(None, n_actions, self.hiddens)
            self.var_param = chainer.Parameter(initializer=var_param_init,
                                               shape=(1,))
            # self.var_param = chainer.Parameter(
            #     initializer=var_param_init, shape=(n_actions,))  # independent

    def __call__(self, s):
        h = s
        for l in self.conv_layers:
            h = self.activation(l(h))
        mean = F.tanh(self.a_stream(h))
        var = F.broadcast_to(self.var_param, mean.shape)
        # var = F.broadcast_to(self.var_func(self.var_param), mean.shape)
        return chainerrl.distribution.GaussianDistribution(mean, var)


class ActorTRPONetForMultiDimensionalSoftmax(chainer.Chain):
    def __init__(self, action_space, n_input_channels=4, activation=F.relu,
                 bias=0.1, var_param_init=0, hiddens=None):
        n_actions = action_space.high + 1
        self.n_input_channels = n_input_channels
        self.activation = activation
        self.hiddens = [512] if hiddens is None else hiddens

        super(ActorTRPONetForMultiDimensionalSoftmax, self).__init__()
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

    def __call__(self, s):
        h = s
        for l in self.conv_layers:
            h = self.activation(l(h))
        for l in self.hidden_layers:
            h = self.activation(l(h))
        out = [layer(h) for layer in self.action_layers]
        return MultiDimensionalSoftmaxDistribution(out)


class ActorVFunc(chainer.Chain):
    def __init__(self, n_input_channels=4, activation=F.relu,
                 bias=0.1, hiddens=None):
        self.n_input_channels = n_input_channels
        self.activation = activation
        self.hiddens = [512] if hiddens is None else hiddens

        super(ActorVFunc, self).__init__()
        with self.init_scope():
            self.conv_layers = chainer.ChainList(
                L.Convolution2D(n_input_channels, 32, 8, stride=4,
                                initial_bias=bias),
                L.Convolution2D(32, 64, 4, stride=2, initial_bias=bias),
                L.Convolution2D(64, 64, 3, stride=1, initial_bias=bias))
            self.a_stream = chainerrl.links.mlp.MLP(None, 1, self.hiddens)

    def __call__(self, s):
        h = s
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
                 bias=0.1, action_wrapper=None, hiddens=None):
        self.n_input_channels = n_input_channels
        self.activation = activation
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
            self.a_stream = chainerrl.links.mlp.MLP(
                None, 1, self.hiddens, nonlinearity=self.activation)

    def __call__(self, s, a):
        num_data = s.shape[0]
        h = s
        for l in self.conv_layers:
            h = self.activation(l(h))
        # flatten
        h = F.reshape(h, (num_data, -1))
        if self.action_wrapper == 'discrete':
            a = a.reshape(-1, 1)
        return self.a_stream(F.concat((h, a), axis=1))
