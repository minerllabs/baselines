"""
MIT License

Copyright (c) Preferred Networks, Inc.
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

from logging import getLogger
import numpy as np
from chainerrl.wrappers.atari_wrappers import LazyFrames
from collections import deque

logger = getLogger(__name__)


def _skip_frames(orig_obs, orig_action, orig_reward, orig_next_obs, orig_done,
                 frameskip):
    orig_length = len(orig_action)
    for param in [orig_obs, orig_action, orig_reward,
                  orig_next_obs, orig_done]:
        assert orig_length == len(param)

    obs = []
    action = []
    reward = []
    next_obs = []
    done = []
    for i in range(orig_length):
        if i % frameskip:
            # skip
            for idx, value in enumerate(orig_action[i]):
                if abs(action[-1][idx]) < abs(value):
                    action[-1][idx] = value
            reward[-1] += orig_reward[i]
            next_obs[-1] = orig_next_obs[i]
            done[-1] = orig_done[i]
            continue
        else:
            obs.append(orig_obs[i])
            action.append(orig_action[i])
            reward.append(orig_reward[i])
            next_obs.append(orig_next_obs[i])
            done.append(orig_done[i])
    return (
        np.array(obs), np.array(action), np.array(reward),
        np.array(next_obs), np.array(done))


def _stack_frames(orig_obs, orig_next_obs, framestack):
    '''
    This implementation is based on
    https://github.com/chainer/chainerrl/blob/master/chainerrl/wrappers/atari_wrappers.py#L183
    '''
    def get_lazyframes(obs, framestack):
        ob = deque([], maxlen=framestack)
        ret = []
        for _ in range(framestack):
            ob.append(obs[0])
        for current_ob in obs:
            ob.append(current_ob)
            ret.append(LazyFrames(list(ob), stack_axis=0))
        return np.array(ret, dtype=LazyFrames)

    return (get_lazyframes(orig_obs, framestack),
            get_lazyframes(orig_next_obs, framestack))


class ExpertDataset:
    '''
    Dataset controller which converts original observations and actions
        to ones baseline agents use.
    Data can be loaded by using sample() function.

    Parameter
    ---------
    original_dataset
        An original minerl dataset. It assumes that original_dataset.sarsd_iterator is callable.
    observation_converter
        Converter of original observations to desidered ones. Examples are in `observation_wrappers.py`.
    action_converter
        Converter of original actions to desidered ones. Examples are in `action_wrappers.py`.
    frameskip
        Frameskips.
    framestack
        Framestacks. It returns LazyFrames when framestack > 1.
    shuffle
        If `False`, it returns every states on a sequential order.
    '''
    def __init__(self, original_dataset, observation_converter=(lambda x: x),
                 action_converter=(lambda x: x), frameskip=1,
                 framestack=1, shuffle=True):
        self.observation_converter = observation_converter
        self.action_converter = action_converter
        self.obs = None
        self.action = None
        self.reward = None
        self.next_obs = None
        self.done = None
        self.size = 0
        self.frameskip = frameskip
        self.framestack = framestack
        self.shuffle = shuffle
        self._convert(original_dataset)
        self.indices = []
        self._reset_indices()

    def _convert(self, original_dataset):
        for (orig_obs, orig_action, orig_reward, orig_next_obs, orig_done) \
                in original_dataset.sarsd_iter(1, -1):
            obs = self.observation_converter(orig_obs)
            action = self.action_converter(orig_action)
            next_obs = self.observation_converter(orig_next_obs)
            obs, action, reward, next_obs, done = _skip_frames(
                obs, action, orig_reward, next_obs, orig_done, self.frameskip)

            if self.framestack > 1:
                obs, next_obs = _stack_frames(obs, next_obs, self.framestack)

            self.obs = (
                obs if self.obs is None else
                np.concatenate((self.obs, obs)))
            self.action = (
                action if self.action is None else
                np.concatenate((self.action, action)))
            self.reward = (
                reward if self.reward is None else
                np.concatenate((self.reward, reward)))
            self.next_obs = (
                next_obs if self.next_obs is None else
                np.concatenate((self.next_obs, next_obs)))
            self.done = (
                done if self.done is None else
                np.concatenate((self.done, done)))
        self.size = len(self.obs)

    def _reset_indices(self):
        if self.shuffle:
            self.indices = np.random.permutation(np.arange(self.size))
        else:
            self.indices = np.arange(self.size)

    def sample(self):
        index = self.indices[0]
        obs = self.obs[index]
        action = self.action[index]
        reward = self.reward[index]
        next_obs = self.next_obs[index]
        done = self.done[index]
        self.indices = np.delete(self.indices, 0)
        if len(self.indices) == 0:
            self._reset_indices()
        return obs, action, reward, next_obs, done
