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
from observation_wrappers import (
    generate_pov_converter, generate_pov_with_compass_converter,
    generate_unified_observation_converter)


def test_generate_pov_converter():
    original_obs = {'pov': np.array([
        [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]],
        [[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]],
    ])}
    expected_obs = np.array([
        [[[1, 0], [0, 1]], [[0, 1], [0, 1]], [[0, 0], [1, 1]]],
        [[[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]]
    ])
    converter = generate_pov_converter()
    obs = converter(original_obs)
    assert np.allclose(expected_obs, obs)


def test_generate_pov_converter_with_grayscale():
    original_obs = {'pov': np.array([
        [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]],
        [[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]],
    ])}
    expected_obs = np.array([
        [[[0.299, 0.587], [0.114, 1]]],
        [[[0, 0], [0, 0]]]
    ])
    converter = generate_pov_converter(grayscale=True)
    obs = converter(original_obs)
    assert np.allclose(expected_obs, obs)


def test_generate_unified_observation_converter():
    converter = generate_unified_observation_converter(region_size=1)

    original_obs = {
        'pov': np.array([
            [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]],
            [[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]],
        ]),
        'compassAngle': [-90, 180],
        'inventory': OrderedDict([('aaa', [0, 2]), ('bbb', [1, 3]), ('ccc', [1, 1])]),
    }
    expected_obs = np.array([
        [[[1, 0], [0, 1]], [[0, 1], [0, 1]], [[0, 0], [1, 1]],
         [[-0.5, -0.5], [-0.5, -0.5]],
         [[0, 1 / 2], [1 / 2, 0]]],
        [[[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]],
         [[1, 1], [1, 1]],
         [[2 / 3, 3 / 4], [1 / 2, 0]]],
    ])
    obs = converter(original_obs)
    assert np.allclose(expected_obs, obs)

    original_obs = {
        'pov': np.array([
            [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]],
            [[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]],
        ]),
    }
    expected_obs = np.array([
        [[[1, 0], [0, 1]], [[0, 1], [0, 1]], [[0, 0], [1, 1]]],
        [[[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]],
    ])
    obs = converter(original_obs)
    assert np.allclose(expected_obs, obs)


def test_generate_pov_with_compass_converter():
    original_obs = {
        'pov': np.array([
            [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]],
            [[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]],
        ]),
        'compassAngle': [-90, 180]}
    expected_obs = np.array([
        [[[1, 0], [0, 1]], [[0, 1], [0, 1]], [[0, 0], [1, 1]],
         [[-0.5, -0.5], [-0.5, -0.5]]],
        [[[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]],
         [[1, 1], [1, 1]]]
    ])
    converter = generate_pov_with_compass_converter()
    obs = converter(original_obs)
    assert np.allclose(expected_obs, obs)


def test_generate_pov_with_compass_converter_with_grayscale():
    original_obs = {
        'pov': np.array([
            [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]],
            [[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]],
        ]),
        'compassAngle': [-90, 180]}
    expected_obs = np.array([
        [[[0.299, 0.587], [0.114, 1]], [[-0.5, -0.5], [-0.5, -0.5]]],
        [[[0, 0], [0, 0]], [[1, 1], [1, 1]]]
    ])
    converter = generate_pov_with_compass_converter(grayscale=True)
    obs = converter(original_obs)
    assert np.allclose(expected_obs, obs)


def test_generate_unified_observation_converter_with_grayscale():
    converter = generate_unified_observation_converter(region_size=1, grayscale=True)

    original_obs = {
        'pov': np.array([
            [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]],
            [[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]],
        ]),
        'compassAngle': [-90, 180],
        'inventory': OrderedDict([('aaa', [0, 2]), ('bbb', [1, 3]), ('ccc', [1, 1])]),
    }
    expected_obs = np.array([
        [[[0.299, 0.587], [0.114, 1]],
         [[-0.5, -0.5], [-0.5, -0.5]],
         [[0, 1 / 2], [1 / 2, 0]]],
        [[[0, 0], [0, 0]],
         [[1, 1], [1, 1]],
         [[2 / 3, 3 / 4], [1 / 2, 0]]],
    ])
    obs = converter(original_obs)
    assert np.allclose(expected_obs, obs)

    original_obs = {
        'pov': np.array([
            [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]],
            [[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]],
        ]),
    }
    expected_obs = np.array([
        [[[0.299, 0.587], [0.114, 1]]],
        [[[0, 0], [0, 0]]]
    ])
    obs = converter(original_obs)
    assert np.allclose(expected_obs, obs)
