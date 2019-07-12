import numpy as np

import chainer
from chainer import functions as F
from chainer import links as L

import chainerrl


class A3CFF(chainer.ChainList, chainerrl.agents.a3c.A3CModel):
    def __init__(self, n_actions, head):
        self.head = head
        self.pi = chainerrl.policy.FCSoftmaxPolicy(self.head.n_output_channels, n_actions)
        self.v = chainerrl.v_function.FCVFunction(self.head.n_output_channels)
        super().__init__(self.head, self.pi, self.v)

    def pi_and_v(self, state):
        out = self.head(state)
        return self.pi(out), self.v(out)


class NatureDQNHead(chainer.ChainList):
    """DQN's head (Nature version)"""
    def __init__(self, n_input_channels=4, n_output_channels=512, activation=F.relu, bias=0.1):
        self.n_input_channels = n_input_channels
        self.activation = activation
        self.n_output_channels = n_output_channels

        layers = [
            L.Convolution2D(n_input_channels, 32, 8, stride=4, initial_bias=bias),
            L.Convolution2D(32, 64, 4, stride=2, initial_bias=bias),
            L.Convolution2D(64, 64, 3, stride=1, initial_bias=bias),
            # the shape of resulting future map is 4x4x64=1024 when the first conv input's size is 64x64.
            # NOTE: we use `None` as `in_size` to defer the parameter initialization for usability,
            # but the size should be checked with concrete input_size (for example, 1024.)
            # L.Linear(3136, n_output_channels, initial_bias=bias),
            L.Linear(None, n_output_channels, initial_bias=bias),
        ]

        super(NatureDQNHead, self).__init__(*layers)

    def __call__(self, state):
        h = state
        for layer in self:
            h = self.activation(layer(h))
        return h


class DuelingDQN(chainer.Chain, chainerrl.q_function.StateQFunction):
    """Dueling Q-Network
    See: http://arxiv.org/abs/1511.06581
    """
    def __init__(self, n_actions, n_input_channels=4, activation=F.relu, bias=0.1, hiddens=None):
        self.n_actions = n_actions
        self.n_input_channels = n_input_channels
        self.activation = activation
        self.hiddens = [512] if hiddens is None else hiddens

        super().__init__()
        with self.init_scope():
            self.conv_layers = chainer.ChainList(
                L.Convolution2D(n_input_channels, 32, 8, stride=4, initial_bias=bias),
                L.Convolution2D(32, 64, 4, stride=2, initial_bias=bias),
                L.Convolution2D(64, 64, 3, stride=1, initial_bias=bias))

            # https://fomoro.com/research/article/receptive-field-calculator#8,4,1,VALID;4,2,1,VALID;3,1,1,VALID
            # the shape of resulting future map is 4x4x64=1024 when the first conv input's size is 64x64.
            # NOTE: we use `None` as `in_size` to defer the parameter initialization for usability,
            # but the size should be checked with concrete input_size (for example, 1024.)
            # self.a_stream = chainerrl.links.mlp.MLP(3136, n_actions, self.hiddens)
            self.a_stream = chainerrl.links.mlp.MLP(None, n_actions, self.hiddens)
            self.v_stream = chainerrl.links.mlp.MLP(None, 1, self.hiddens)

    def __call__(self, x):
        h = x
        for l in self.conv_layers:
            h = self.activation(l(h))

        # Advantage
        batch_size = x.shape[0]
        ya = self.a_stream(h)
        mean = F.reshape(F.sum(ya, axis=1) / self.n_actions, (batch_size, 1))
        ya, mean = F.broadcast(ya, mean)
        ya -= mean

        # State value
        ys = self.v_stream(h)

        ya, ys = F.broadcast(ya, ys)
        q = ya + ys
        return chainerrl.action_value.DiscreteActionValue(q)


class DistributionalDuelingDQN(
        chainer.Chain, chainerrl.q_function.StateQFunction, chainerrl.recurrent.RecurrentChainMixin):
    """Distributional dueling fully-connected Q-function with discrete actions."""
    def __init__(self, n_actions, n_atoms, v_min, v_max,
                 n_input_channels=4, activation=F.relu, bias=0.1):
        assert n_atoms >= 2
        assert v_min < v_max

        self.n_actions = n_actions
        self.n_input_channels = n_input_channels
        self.activation = activation
        self.n_atoms = n_atoms

        super().__init__()
        z_values = self.xp.linspace(v_min, v_max, num=n_atoms, dtype=np.float32)
        self.add_persistent('z_values', z_values)

        with self.init_scope():
            self.conv_layers = chainer.ChainList(
                L.Convolution2D(n_input_channels, 32, 8, stride=4, initial_bias=bias),
                L.Convolution2D(32, 64, 4, stride=2, initial_bias=bias),
                L.Convolution2D(64, 64, 3, stride=1, initial_bias=bias))

            # https://fomoro.com/research/article/receptive-field-calculator#8,4,1,VALID;4,2,1,VALID;3,1,1,VALID
            # the shape of resulting future map is 4x4x64=1024 when the first conv input's size is 64x64.
            # NOTE: we use `None` as `in_size` to defer the parameter initialization for usability,
            # but the size should be checked with concrete input_size (for example, 1024.)
            # self.main_stream = L.Linear(3136, 1024)
            self.main_stream = L.Linear(None, 1024)
            self.a_stream = L.Linear(512, n_actions * n_atoms)
            self.v_stream = L.Linear(512, n_atoms)

    def __call__(self, x):
        h = x
        for l in self.conv_layers:
            h = self.activation(l(h))

        # Advantage
        batch_size = x.shape[0]

        h = self.activation(self.main_stream(h))
        h_a, h_v = F.split_axis(h, 2, axis=-1)
        ya = F.reshape(self.a_stream(h_a), (batch_size, self.n_actions, self.n_atoms))

        mean = F.sum(ya, axis=1, keepdims=True) / self.n_actions

        ya, mean = F.broadcast(ya, mean)
        ya -= mean

        # State value
        ys = F.reshape(self.v_stream(h_v), (batch_size, 1, self.n_atoms))
        ya, ys = F.broadcast(ya, ys)
        q = F.softmax(ya + ys, axis=2)

        return chainerrl.action_value.DistributionalDiscreteActionValue(q, self.z_values)
