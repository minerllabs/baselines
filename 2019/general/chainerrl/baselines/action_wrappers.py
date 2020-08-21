"""
MIT License

Copyright (c) Preferred Networks, Inc.
"""

from collections import OrderedDict
import copy

import numpy as np


BINARY_KEYS = ['forward', 'back', 'left', 'right', 'jump', 'sneak', 'sprint', 'attack']
ENUM_KEYS = ['place', 'equip', 'craft', 'nearbyCraft', 'nearbySmelt']
ALL_ORDERED_KEYS = BINARY_KEYS + ['camera'] + ENUM_KEYS
# Number of actions for each enum elements on each environment
NUM_ENUM_ACTIONS = {
    'MineRLTreechop-v0': {},
    'MineRLNavigate-v0': OrderedDict([
        ('place', 2),
    ]),
    'MineRLNavigateExtreme-v0': OrderedDict([
        ('place', 2),
    ]),
    'MineRLObtainIronPickaxe-v0': OrderedDict([
        ('place', 7),
        ('equip', 7),
        ('craft', 5),
        ('nearbyCraft', 8),
        ('nearbySmelt', 3),
    ]),
    'MineRLObtainDiamond-v0': OrderedDict([
        ('place', 7),
        ('equip', 7),
        ('craft', 5),
        ('nearbyCraft', 8),
        ('nearbySmelt', 3),
    ]),
}


class DiscreteActionConverter:
    '''
    Converter from original action dict to a discrete action.
    This converter is consisted of a sequence of queries.

    Parameters
    ----------
    query (list of tuple): conditions of selecting each action.
        each tuple consisted 4 elements: query_type, condition_parameter, referring element, and return
        query_type is either less, more, or same.
        When the value of referring element satisfies a pair of query_type and condition_parameter,
        then the action speficied by return is taken.
        If multiple queries are satisfied, only first one is used.
        If none of multiple queries is satisfied. Return 0 as noop.
    '''
    def __init__(self, query):
        self.query = query

    def _comp(self, value_a, value_b, operator):
        if operator == 'less':
            return value_a < value_b
        elif operator == 'more':
            return value_a > value_b
        elif operator == 'same':
            return abs(value_a - value_b) < 1e-3
        else:
            raise ValueError('Unknown operator {}'.format(operator))

    def __call__(self, action):
        episode_len = len(action['forward'])
        ret = np.zeros((episode_len, 1), dtype=np.int32)
        for operator, comp, action_name, value in reversed(self.query):
            for idx in range(episode_len):
                if action_name == 'camera_pitch':
                    comp_action = action['camera'][idx, 0]
                elif action_name == 'camera_yaw':
                    comp_action = action['camera'][idx, 1]
                else:
                    comp_action = action[action_name][idx]
                if self._comp(comp_action, comp, operator):
                    ret[idx][0] = value
        # noop
        return ret


def generate_mapping(env_name, actions, num_camera_discretize,
                     allow_pitch=False, exclude_noop=False):
    '''
    Generate a mapping function from action names to indices.
    '''
    discrete_indices = {}
    agent_index = 0 if exclude_noop else 1
    for key in actions:
        discrete_indices[key] = agent_index
        if key == 'camera':
            agent_index += num_camera_discretize - 1
            if allow_pitch:
                agent_index += num_camera_discretize - 1
        elif key in NUM_ENUM_ACTIONS[env_name]:
            agent_index += NUM_ENUM_ACTIONS[env_name][key] - 1
        else:
            # BINARY_KEYS
            agent_index += 1

    return discrete_indices


def generate_priority(original_order, prioritized_elements=None):
    '''
    Generate an order of queries.
    '''
    if prioritized_elements is not None:
        priorities = copy.copy(prioritized_elements)
        for key in original_order:
            if key in prioritized_elements:
                continue
            priorities.append(key)
    else:
        priorities = copy.copy(original_order)
    return priorities


def generate_query(env_name, priorities, reverse_keys, discrete_indices,
                   max_camera_range, num_camera_discretize, allow_pitch=False):
    '''
    Generate queries.
    '''
    query = []
    for key in priorities:
        if key == 'camera':
            # yaw
            assert num_camera_discretize % 2 == 1
            half_n = num_camera_discretize // 2
            scale = max_camera_range * 2 / (num_camera_discretize - 1)
            for i in range(half_n):
                query.append(('less', -max_camera_range + scale * (0.5 + i),
                              'camera_yaw',
                              discrete_indices[key] + i))
                query.append(('more', max_camera_range - scale * (0.5 + i),
                              'camera_yaw',
                              discrete_indices[key] + num_camera_discretize - 2 - i))
            if allow_pitch:
                # pitch
                for i in range(half_n):
                    query.append(('less', -max_camera_range + scale * (0.5 + i),
                                  'camera_pitch',
                                  discrete_indices[key] + num_camera_discretize - 1 + i))
                    query.append(('more', max_camera_range - scale * (0.5 + i),
                                  'camera_pitch',
                                  discrete_indices[key] + num_camera_discretize * 2 - 3 - i))
        elif key in NUM_ENUM_ACTIONS[env_name]:
            n = NUM_ENUM_ACTIONS[env_name][key]
            for i in range(1, n):
                query.append(('same', i,
                              key, discrete_indices[key] + i - 1))
        elif key in BINARY_KEYS:
            # BINARY_KEYS
            if key in reverse_keys:
                query.append(('less', 0.5,
                              key, discrete_indices[key]))
            else:
                query.append(('more', 0.5,
                              key, discrete_indices[key]))
        else:
            raise ValueError('Unknown key {}.'.format(key))
    return query


def generate_discrete_converter(env_name,
                                prioritized_elements=None,
                                always_keys=None,
                                reverse_keys=None,
                                exclude_keys=None,
                                exclude_noop=False,
                                allow_pitch=False,
                                max_camera_range=10,
                                num_camera_discretize=3):
    '''
    Generate a DiscreteActionConverter for expert dataset.

    Parameters
    ----------
    env_name
        Environment name.
    prioritized_elements
        Action names contained in this list are considered first.
        Then the remaining actions are considered.
    always_keys
        List of action keys, which should be always pressed throughout interaction with environment.
        If specified, the "noop" action is also affected.
    reverse_keys
        List of action keys, which should be always pressed but can be turn off via action.
        If specified, the "noop" action is also affected.
    exclude_keys
        List of action keys, which should be ignored for discretizing action space.
    exclude_noop
        The "noop" will be excluded from discrete action list.
    allow_pitch
        If it is true, it enables a converter to take pitch control action.
    num_camera_discretize
        Number of discretization of yaw control (must be odd).
    max_camera_range
        Maximum value of yaw control.

    '''
    env_name = env_name.replace("Dense", "")

    prioritized_elements = [] if prioritized_elements is None else prioritized_elements
    always_keys = [] if always_keys is None else always_keys
    reverse_keys = [] if reverse_keys is None else reverse_keys
    exclude_keys = [] if exclude_keys is None else exclude_keys

    # Set indices on discrete actions
    actions = []
    for key in BINARY_KEYS + ['camera'] + list(NUM_ENUM_ACTIONS[env_name]):
        if (key not in (always_keys + exclude_keys)):
            actions.append(key)

    discrete_indices = generate_mapping(
        env_name, actions, num_camera_discretize, allow_pitch, exclude_noop)

    # Set priorities
    _original_order = [
        'nearbyCraft', 'nearbySmelt', 'craft', 'equip', 'place', 'camera',
        'forward', 'back', 'left', 'right', 'jump', 'sneak', 'sprint',
        'attack']
    original_order = []
    for key in _original_order:
        if key in actions:
            original_order.append(key)

    # sanity check
    for key in prioritized_elements:
        if key not in actions:
            raise ValueError('Unknown key {} in prioritized_elements.'.format(key))

    priorities = generate_priority(original_order, prioritized_elements)

    # sanity check
    for key in priorities:
        if key not in actions:
            raise ValueError('Unknown key {} in priorities.'.format(key))

    # Set query
    query = generate_query(
        env_name, priorities, reverse_keys, discrete_indices,
        max_camera_range, num_camera_discretize, allow_pitch)

    return DiscreteActionConverter(query)


def generate_continuous_converter(env_name, allow_pitch, max_camera_range):
    '''
    Generate a continuous action converter for expert dataset.

    Parameters
    ----------
    env_name
        Environment name.
    allow_pitch
        If it is true, it enables a converter to take pitch control action.
    max_camera_range
        Maximum value of yaw control.
    '''

    env_name = env_name.replace("Dense", "")

    def action_converter(action):
        episode_len = len(action['forward'])
        value = np.zeros((episode_len, len(action) + 1), dtype=np.float32)
        idx = 0
        for key in ALL_ORDERED_KEYS:
            if key not in action:
                continue
            if key in BINARY_KEYS:
                value[:, idx] = -1 + action[key] * 2
                idx += 1
            elif key == 'camera':
                if allow_pitch:
                    value[:, idx] = np.clip(action[key][:, 0] / max_camera_range,
                                            -1, 1)
                value[:, idx + 1] = np.clip(action[key][:, 1] / max_camera_range,
                                            -1, 1)
                idx += 2
            else:
                value[:, idx] = -1 + action[key] / (NUM_ENUM_ACTIONS[env_name][key] - 1) * 2
                idx += 1

        return value

    return action_converter


def generate_multi_dimensional_softmax_converter(
        allow_pitch, max_camera_range, num_camera_discretize):
    '''
    Generate a multi dimensional discrete action converter for expert dataset.

    Parameters
    ----------
    allow_pitch
        If it is true, it enables a converter to take pitch control action.
    num_camera_discretize
        Number of discretization of yaw control (must be odd).
    max_camera_range
        Maximum value of yaw control.
    '''

    def action_converter(action):
        episode_len = len(action['forward'])
        value = np.zeros((episode_len, len(action) + 1), dtype=np.int32)
        idx = 0
        for key in ALL_ORDERED_KEYS:
            if key not in action:
                continue
            if key == 'camera':
                if allow_pitch:
                    scaled = (
                        (action[key][:, 0] + max_camera_range)
                        * (num_camera_discretize - 1)
                        / (max_camera_range * 2))
                    value[:, idx] = np.clip(np.round(scaled),
                                            0, num_camera_discretize - 1)
                else:
                    value[:, idx] = np.ones_like(value[:, idx], dtype=np.int32) * (num_camera_discretize - 1) // 2
                scaled = (
                    (action[key][:, 1] + max_camera_range)
                    * (num_camera_discretize - 1)
                    / (max_camera_range * 2))
                value[:, idx + 1] = np.clip(np.round(scaled),
                                            0, num_camera_discretize - 1)
                idx += 2
            else:
                value[:, idx] = action[key]
                idx += 1

        return value

    return action_converter
