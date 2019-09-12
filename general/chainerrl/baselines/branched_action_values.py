from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

from cached_property import cached_property
from chainer import functions as F

from chainerrl.action_value import ActionValue


class BranchedActionValue(ActionValue):
    """Q-function output for a branched action space.
    Args:
        branches (list):
            Each element of the list is a Q-function for an action dimension
    """

    def __init__(self, branches, q_values_formatter=lambda x: x):
        self.branches = branches
        self.q_values_formatter = q_values_formatter

    @cached_property
    def greedy_actions(self):
        actions = []

        for branch in self.branches:
            actions.append(branch.q_values.array.argmax(axis=1).reshape(-1, 1))

        return F.hstack(actions)

    @cached_property
    def max(self):
        chosen_q_values = []

        for branch in self.branches:
            chosen_q_values.append(branch.max.reshape(-1, 1))

        return F.hstack(chosen_q_values)

    def evaluate_actions(self, actions):
        branch_q_values = []

        for i, branch in enumerate(self.branches):
            branch_actions = actions[:, i]
            branch_q_values.append(branch.evaluate_actions(
                branch_actions).reshape(-1, 1))

        return F.hstack(branch_q_values)

    @property
    def params(self):
        branch_params = []

        for branch in self.branches:
            branch_params.extend(list(branch.params))

        return tuple(branch_params)
