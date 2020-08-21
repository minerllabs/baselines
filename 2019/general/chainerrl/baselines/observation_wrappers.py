"""
MIT License

Copyright (c) Preferred Networks, Inc.
"""
import copy
import numpy as np
import cv2


def generate_pov_converter(grayscale=False):
    '''
    Observation converter for pov only observations.
    '''
    def converter(observation):
        observation = observation['pov'].astype(np.float32)
        ret = []
        for orig_obs in observation:
            obs = copy.deepcopy(orig_obs)
            if grayscale:
                obs = np.expand_dims(
                    cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY), axis=-1)
            obs = obs / 255
            obs = np.moveaxis(obs, [0, 1, 2], [1, 2, 0])
            ret.append(obs)
        return np.array(ret)

    return converter


def generate_pov_with_compass_converter(grayscale=False):
    '''
    Observation converter for pov and compassAngle observations such as `Navigate` tasks.
    '''
    def converter(observation):
        ret = []
        for pov, compass_angle in zip(
                observation['pov'].astype(np.float32),
                observation['compassAngle']):
            obs = pov
            if grayscale:
                obs = np.expand_dims(
                    cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY), axis=-1)
            obs = obs / 255
            compass_angle_scale = 180
            compass_scaled = compass_angle / compass_angle_scale
            compass_channel = np.ones(shape=list(obs.shape[:-1]) + [1], dtype=pov.dtype) * compass_scaled
            obs = np.concatenate([obs, compass_channel], axis=-1)
            obs = np.moveaxis(obs, [0, 1, 2], [1, 2, 0])
            ret.append(obs)
        return np.array(ret)

    return converter


def generate_unified_observation_converter(grayscale=False, region_size=8):
    '''
    Observation converter for all of pov, compassAngle, and inventory.
    '''
    def converter(observation):
        ret = []
        batch_size = len(observation['pov'])
        compassAngles = observation['compassAngle'] if 'compassAngle' in observation else [None] * batch_size
        for idx, (pov, compass_angle) in enumerate(zip(
                observation['pov'].astype(np.float32),
                compassAngles)):
            obs = pov
            if grayscale:
                obs = np.expand_dims(
                    cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY), axis=-1)
            obs = obs / 255

            if compass_angle is not None:
                compass_angle_scale = 180
                compass_scaled = compass_angle / compass_angle_scale
                compass_channel = np.ones(shape=list(obs.shape[:-1]) + [1], dtype=pov.dtype) * compass_scaled
                obs = np.concatenate([obs, compass_channel], axis=-1)

            if 'inventory' in observation:
                assert len(obs.shape[:-1]) == 2
                region_max_height = obs.shape[0]
                region_max_width = obs.shape[1]
                rs = region_size
                if min(region_max_height, region_max_width) < rs:
                    raise ValueError("'region_size' is too large.")
                num_element_width = region_max_width // rs
                inventory_channel = np.zeros(shape=list(obs.shape[:-1]) + [1], dtype=pov.dtype)
                for key_idx, key in enumerate(observation['inventory'].keys()):
                    item_scaled = np.clip(1 - 1 / (observation['inventory'][key][idx] + 1),  # Inversed
                                          0, 1)
                    item_channel = np.ones(shape=[rs, rs, 1], dtype=pov.dtype) * item_scaled
                    width_low = (key_idx % num_element_width) * rs
                    height_low = (key_idx // num_element_width) * rs
                    if height_low + rs > region_max_height:
                        raise ValueError("Too many elements on 'inventory'. Please decrease 'region_size' of each component.")
                    inventory_channel[height_low:(height_low + rs), width_low:(width_low + rs), :] = item_channel
                obs = np.concatenate([obs, inventory_channel], axis=-1)

            obs = np.moveaxis(obs, [0, 1, 2], [1, 2, 0])
            ret.append(obs)
        return np.array(ret)

    return converter
