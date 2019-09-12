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
from collections import OrderedDict
from action_wrappers import (
    DiscreteActionConverter, generate_mapping, generate_priority, generate_query,
    generate_discrete_converter, generate_continuous_converter,
    generate_multi_dimensional_softmax_converter)


def test_action_converter():
    query = [
        ['more', 2, 'forward', 1],
        ['less', 0, 'forward', 2],
        ['same', 1, 'forward', 3],
        ['same', 1, 'camera_yaw', 4],
        ['same', 1, 'camera_pitch', 5],
        ['same', 1.5, 'forward', 6],
    ]
    converter = DiscreteActionConverter(query)
    expert_actions = OrderedDict([
        ('forward', np.array([3, -1, 1, 1.5, 1.5, 1.5, 0])),
        ('camera', np.array(
            [[0, 0], [0, 0], [0, 0], [1, 1], [1, 0], [0, 0], [0, 0]],
            dtype=np.float32))
    ])
    expected_actions = np.array([[1], [2], [3], [4],
                                 [5], [6], [0]], dtype=np.int32)
    converted = converter(expert_actions)
    assert np.allclose(expected_actions, converted)


def test_generate_agent_mapping():
    env_name = 'MineRLObtainDiamond-v0'
    actions = ['craft', 'camera', 'attack', 'forward']
    discrete_indices_expected = {'craft': 1, 'camera': 5,
                                 'attack': 7, 'forward': 8}
    discrete_indices = generate_mapping(
        env_name, actions, num_camera_discretize=3)
    assert discrete_indices_expected == discrete_indices

    discrete_indices_expected = {'craft': 1, 'camera': 5,
                                 'attack': 9, 'forward': 10}
    discrete_indices = generate_mapping(
        env_name, actions, num_camera_discretize=3, allow_pitch=True)
    assert discrete_indices_expected == discrete_indices

    discrete_indices_expected = {'craft': 0, 'camera': 4,
                                 'attack': 6, 'forward': 7}
    discrete_indices = generate_mapping(
        env_name, actions, num_camera_discretize=3, exclude_noop=True)
    assert discrete_indices_expected == discrete_indices


def test_generate_priority():
    original_order = ['aaa', 'bbb', 'ccc']
    expected_order = ['aaa', 'bbb', 'ccc']
    priorities = generate_priority(original_order)
    assert expected_order == priorities
    expected_order = ['ccc', 'bbb', 'aaa']
    priorities = generate_priority(original_order, ['ccc', 'bbb'])
    assert expected_order == priorities


def test_generate_query():
    env_name = 'MineRLObtainDiamond-v0'
    priorities = ['craft', 'camera', 'forward', 'left']
    reverse_keys = ['forward']
    discrete_indices = {'craft': 1, 'camera': 5, 'forward': 9, 'left': 10}
    query = generate_query(env_name, priorities, reverse_keys,
                           discrete_indices, allow_pitch=True,
                           max_camera_range=10, num_camera_discretize=3)
    query_expected = [
        ('same', 1, 'craft', 1),
        ('same', 2, 'craft', 2),
        ('same', 3, 'craft', 3),
        ('same', 4, 'craft', 4),
        ('less', -5.0, 'camera_yaw', 5),
        ('more', 5.0, 'camera_yaw', 6),
        ('less', -5.0, 'camera_pitch', 7),
        ('more', 5.0, 'camera_pitch', 8),
        ('less', 0.5, 'forward', 9),
        ('more', 0.5, 'left', 10),
    ]
    assert len(query_expected) == len(query)
    for expected, generated in zip(query_expected, query):
        for i in range(4):
            assert expected[i] == generated[i]


def test_generate_discrete_converter():
    env_name = 'MineRLNavigateDense-v0'
    converter = generate_discrete_converter(
        env_name,
        prioritized_elements=['attack'],
        always_keys=['forward', 'jump', 'sprint'],
        reverse_keys=['attack'],
        exclude_keys=['back', 'sneak'])

    # wrapper = generate_wrapper_function_to_multidimensional_softmax_action()

    # discrete actions: [noop, left, right, attack, camera_yaw0,
    #                    camera_yaw1, place0, place1]
    action = OrderedDict([
        ('forward', np.array([0, 0])),
        ('back', np.array([0, 0])),
        ('left', np.array([0, 0])),
        ('right', np.array([0, 1])),
        ('jump', np.array([0, 0])),
        ('sneak', np.array([0, 0])),
        ('sprint', np.array([0, 0])),
        ('attack', np.array([1, 0])),
        ('camera', np.array([[10, 0], [10, 0]], dtype=np.float32)),
        ('place', np.array([0, 0]))
    ])
    expected = np.array([[0], [3]])
    converted = converter(action)
    assert converted.dtype == np.int32
    assert np.allclose(expected, converted)


def test_generate_continuous_converter():
    env_name = 'MineRLObtainDiamondDense-v0'
    actions = [OrderedDict([
        ('attack', np.array([0, 0])),
        ('forward', np.array([0, 0])),
        ('jump', np.array([0, 0])),
        ('sneak', np.array([0, 0])),
        ('back', np.array([0, 0])),
        ('left', np.array([0, 0])),
        ('right', np.array([1, 1])),
        ('sprint', np.array([1, 1])),
        ('camera', np.array([[5, -5], [5, -5]], dtype=np.float32)),
        ('place', np.array([5, 5])),
        ('equip', np.array([4, 4])),
        ('craft', np.array([3, 3])),
        ('nearbyCraft', np.array([2, 2])),
        ('nearbySmelt', np.array([1, 1])),
    ])]
    converter = generate_continuous_converter(
        env_name, allow_pitch=True, max_camera_range=10)
    expected_all = np.array([
        [[-1, -1, -1, 1, -1, -1, 1, -1, 0.5, -0.5, 2 / 3, 1 / 3, 0.5, -3 / 7, 0]],
        [[-1, -1, -1, 1, 1, -1, 1, -1, 0.5, -0.5, 2 / 3, 1 / 3, 0.5, -3 / 7, 0]],
    ])
    for expected, action in zip(expected_all, actions):
        converted = converter(action)
        assert converted.dtype == np.float32
        assert np.allclose(expected, converted)


def test_generate_multi_dimensional_softmax_converter():
    actions = [OrderedDict([
        ('forward', np.array([0, 0])),
        ('back', np.array([0, 0])),
        ('left', np.array([0, 0])),
        ('jump', np.array([0, 0])),
        ('attack', np.array([0, 0])),
        ('sneak', np.array([0, 0])),
        ('right', np.array([1, 1])),
        ('sprint', np.array([1, 1])),
        ('camera', np.array([[5, -5], [5, -5]], dtype=np.float32)),
        ('place', np.array([5, 5])),
        ('equip', np.array([4, 4])),
        ('craft', np.array([3, 3])),
        ('nearbyCraft', np.array([2, 2])),
        ('nearbySmelt', np.array([1, 1])),
    ])]
    converter = generate_multi_dimensional_softmax_converter(
        allow_pitch=True, max_camera_range=10, num_camera_discretize=5)
    expected_all = np.array([
        [[0, 0, 0, 1, 0, 0, 1, 0, 3, 1, 5, 4, 3, 2, 1]],
        [[0, 0, 0, 1, 1, 0, 1, 0, 3, 1, 5, 4, 3, 2, 1]],
    ])
    for expected, action in zip(expected_all, actions):
        converted = converter(action)
        assert converted.dtype == np.int32
        assert np.allclose(expected, converted)
