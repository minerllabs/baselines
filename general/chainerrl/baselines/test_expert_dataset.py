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

import numpy as np
import copy
from unittest.mock import MagicMock
from chainerrl.wrappers.atari_wrappers import LazyFrames
from expert_dataset import ExpertDataset, _skip_frames, _stack_frames


def test__skip_frames():
    orig_obs = np.random.rand(7, 3, 2, 2).astype(np.float32)
    orig_action = np.array([[0], [1], [2], [-3], [4], [5], [6]])
    orig_reward = np.arange(7)
    orig_next_obs = np.random.rand(7, 3, 2, 2).astype(np.float32)
    orig_done = np.array([False] * 6 + [True])
    obs, action, reward, next_obs, done = _skip_frames(
        orig_obs, orig_action, orig_reward, orig_next_obs, orig_done, 4)

    assert obs.shape == (2, 3, 2, 2)
    for idx, expected in enumerate([0, 4]):
        assert np.allclose(obs[idx], orig_obs[expected])

    assert action.shape == (2, 1)
    for idx, expected in enumerate([[-3], [6]]):
        assert np.allclose(action[idx], expected)

    assert reward.shape == (2,)
    for idx, expected in enumerate([6, 15]):
        assert np.allclose(reward[idx], expected)

    assert next_obs.shape == (2, 3, 2, 2)
    for idx, expected in enumerate([3, 6]):
        assert np.allclose(next_obs[idx], orig_next_obs[expected])

    assert done.shape == (2,)
    for idx, expected in enumerate([False, True]):
        assert np.allclose(done[idx], expected)


def test__stack_frames():
    orig_obs = np.random.rand(5, 3, 2, 2)
    orig_next_obs = np.random.rand(5, 3, 2, 2)
    obs, next_obs = _stack_frames(orig_obs, orig_next_obs, 4)
    assert obs.__class__ == np.ndarray
    assert obs.shape == (5,)
    assert next_obs.__class__ == np.ndarray
    assert next_obs.shape == (5,)
    for orig_array, obtained_array in [(orig_obs, obs),
                                       (orig_next_obs, next_obs)]:
        expected_indices = np.zeros(4, dtype=np.int32)
        for idx, ob in enumerate(obtained_array):
            assert ob.__class__ == LazyFrames
            converted_ob = np.array(ob)
            assert converted_ob.__class__ == np.ndarray
            assert converted_ob.shape == (12, 2, 2)
            assert np.allclose(
                converted_ob,
                np.concatenate(tuple(orig_array[expected_indices]), axis=0))
            expected_indices = np.append(expected_indices[1:], idx + 1)


def test_expert_dataset_sample():
    def observation_converter(observation):
        return np.repeat(observation, 2, axis=1)

    def action_converter(action):
        ret = copy.deepcopy(action)
        ret[:, 0] *= 2
        return ret

    episode_1 = (
        np.array([[0, 1], [1, 2]]),
        np.array([[2], [8]]),
        [10, 20],
        np.array([[2, 3], [3, 4]]),
        [False, False, True]
    )
    episode_2 = (
        np.array([[1, 2], [3, 4], [5, 6]]),
        np.array([[3], [4], [5]]),
        [20, 30, 40],
        np.array([[3, 4], [5, 6], [7, 8]]),
        [False, False, False, True]
    )
    expected = (
        np.array([[0, 0, 1, 1], [1, 1, 2, 2], [5, 5, 6, 6]]),
        np.array([[16], [8], [10]]),
        [30, 50, 40],
        np.array([[3, 3, 4, 4], [5, 5, 6, 6], [7, 7, 8, 8]]),
        [True, False, True]
    )

    def mock_func(x, y):
        yield episode_1
        yield episode_2

    original_data = MagicMock()
    original_data.sarsd_iter = mock_func
    dataset = ExpertDataset(original_data,
                            observation_converter=observation_converter,
                            action_converter=action_converter,
                            frameskip=2,
                            shuffle=False)
    for loop in range(2):
        for i in range(3):
            value = dataset.sample()
            for comp_idx in range(4):
                assert np.allclose(expected[comp_idx][i],
                                   value[comp_idx])
            assert expected[4][i] == value[4]

    # framestack
    dataset = ExpertDataset(original_data,
                            observation_converter=observation_converter,
                            action_converter=action_converter,
                            frameskip=2, framestack=2,
                            shuffle=False)
    expected_indices = np.array([[0, 0], [1, 1], [1, 2]])
    for loop in range(2):
        for i in range(3):
            value = dataset.sample()
            for comp_idx in range(4):
                if comp_idx in [0, 3]:
                    assert value[comp_idx].__class__ == LazyFrames
                    assert np.allclose(
                        np.concatenate(
                            tuple(expected[comp_idx][expected_indices[i]]),
                            axis=0),
                        np.array(value[comp_idx]))
                else:
                    assert np.allclose(expected[comp_idx][i],
                                       value[comp_idx])
            assert expected[4][i] == value[4]
