import os
import cv2
import glob
import time
from collections import deque
from logging import getLogger
import numpy as np

from chainerrl.wrappers.atari_wrappers import LazyFrames


def get_encoded_action(env_name, state, index, frameskip, camera_atomic_actions, max_range_of_camera):
    action = []
    n_frames = len(state['reward'][index:index + frameskip])

    # branch for back/forward/left/right (9 actions)
    action_back_forward = 0
    if state['action_back'][index] == 1 and state['action_forward'][index] == 0:
        action_back_forward = 1
    elif state['action_back'][index] == 0 and state['action_forward'][index] == 1:
        action_back_forward = 2

    action_left_right = 0
    if state['action_left'][index] == 1 and state['action_right'][index] == 0:
        action_left_right = 1
    elif state['action_left'][index] == 0 and state['action_right'][index] == 1:
        action_left_right = 2

    # Encode into the range [0, 8]
    action.append(action_back_forward * 3 + action_left_right)

    # branch for attack/sneak/sprint/jump (16 actions, range [0,15])
    encoded_action = 0
    if state['action_attack'][index] == 1:
        encoded_action = encoded_action * 2 + 1
    else:
        encoded_action *= 2
    if state['action_sneak'][index] == 1:
        encoded_action = encoded_action * 2 + 1
    else:
        encoded_action *= 2
    if state['action_sprint'][index] == 1:
        encoded_action = encoded_action * 2 + 1
    else:
        encoded_action *= 2
    if state['action_jump'][index] == 1:
        encoded_action = encoded_action * 2 + 1
    else:
        encoded_action *= 2
    action.append(encoded_action)

    # branch for camera
    camera0 = state['action_camera'][index:index + n_frames, 0].mean()
    camera0 = max(camera0, -max_range_of_camera)
    camera0 = min(camera0, max_range_of_camera)
    camera0 += max_range_of_camera  # range [0, 2 * max_range_of_camera]

    camera1 = state['action_camera'][index:index + n_frames, 1].mean()
    camera1 = max(camera1, -max_range_of_camera)
    camera1 = min(camera1, max_range_of_camera)
    camera1 += max_range_of_camera  # range [0, 2 * max_range_of_camera]

    # discretization and encoding
    segment_size = 2 * max_range_of_camera / (camera_atomic_actions - 1)
    camera0 = (camera0 + segment_size / 2) // segment_size
    camera1 = (camera1 + segment_size / 2) // segment_size
    action.append(int(camera0))
    action.append(int(camera1))

    # branch for craft/equip/nearbyCraft/nearbySmelt/place
    # only one of these can happen at a time
    if env_name in ['MineRLNavigate-v0', 'MineRLNavigateDense-v0',
                    'MineRLNavigateExtreme-v0', 'MineRLNavigateExtremeDense-v0']:
        assert(state['action_place'][index] >= 0
               and state['action_place'][index] <= 1)
        action.append(state['action_place'][index])
    elif env_name in ['MineRLObtainIronPickaxe-v0', 'MineRLObtainIronPickaxeDense-v0',
                      'MineRLObtainDiamond-v0', 'MineRLObtainDiamondDense-v0']:
        if state['action_craft'][index] > 0:
            assert(state['action_craft'][index] <= 4)
            assert(state['action_equip'][index] == 0)
            assert(state['action_nearbyCraft'][index] == 0)
            assert(state['action_nearbySmelt'][index] == 0)
            assert(state['action_place'][index] == 0)

            action.append(1 + state['action_craft'][index])
        elif state['action_equip'][index] > 0:
            assert(state['action_craft'][index] == 0)
            assert(state['action_equip'][index] <= 7)
            assert(state['action_nearbyCraft'][index] == 0)
            assert(state['action_nearbySmelt'][index] == 0)
            assert(state['action_place'][index] == 0)

            action.append(6 + state['action_equip'][index])
        elif state['action_nearbyCraft'][index] > 0:
            assert(state['action_craft'][index] == 0)
            assert(state['action_equip'][index] == 0)
            assert(state['action_nearbyCraft'][index] <= 7)
            assert(state['action_nearbySmelt'][index] == 0)
            assert(state['action_place'][index] == 0)

            action.append(14 + state['action_nearbyCraft'][index])
        elif state['action_nearbySmelt'][index] > 0:
            assert(state['action_craft'][index] == 0)
            assert(state['action_equip'][index] == 0)
            assert(state['action_nearbyCraft'][index] == 0)
            assert(state['action_nearbySmelt'][index] <= 2)
            assert(state['action_place'][index] == 0)

            action.append(22 + state['action_nearbySmelt'][index])
        elif state['action_place'][index] > 0:
            assert(state['action_craft'][index] == 0)
            assert(state['action_equip'][index] == 0)
            assert(state['action_nearbyCraft'][index] == 0)
            assert(state['action_nearbySmelt'][index] == 0)
            assert(state['action_place'][index] <= 6)

            action.append(25 + state['action_place'][index])
        else:
            action.append(0)

    return action


def fill_buffer(env_name, chosen_dirs, replay_buffer, frameskip, frame_stack,
                camera_atomic_actions, max_range_of_camera, use_full_observation,
                logger=getLogger(__name__)):

    compass_angle_scale = 180 / 255  # From baselines/env_wrappers.py

    if env_name.startswith('MineRLNavigate'):
        use_compass = True
    else:
        use_compass = False

    start = time.time()

    for dr in chosen_dirs:
        logger.info("Player: {}".format(dr))
        dr_start = time.time()
        video_file = os.path.join(dr, 'recording.mp4')
        other_file = os.path.join(dr, 'rendered.npz')

        cap = cv2.VideoCapture(video_file)
        others = np.load(other_file)
        length = others['reward'].shape[0]
        logger.info("Length of rewards array: {}".format(length))
        if env_name.startswith('MineRLNavigate'):
            assert(len(others['observation_compassAngle']) == length + 1)
        if 'observation_inventory' in others:
            assert(len(others['observation_inventory']) == length + 1)

        # set expert observations
        frames = []
        other_obs = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            # (64, 64, 3) -> (3, 64, 64)
            if ret:
                if frame.shape != (64, 64, 3):
                    raise Exception(
                        "Image size is not (64, 64). Dir: {}".format(dr))

                if frame_idx < length + 1:
                    if use_compass and not use_full_observation:
                        compass_scaled = others['observation_compassAngle'][frame_idx] / compass_angle_scale
                        compass_channel = np.ones(shape=list(
                            frame.shape[:-1]) + [1], dtype=frame.dtype) * compass_scaled
                        frame = np.concatenate(
                            [frame, compass_channel], axis=-1)

                    if use_full_observation:
                        obs = []

                        if env_name.startswith('MineRLNavigate'):
                            compass_scaled = others['observation_compassAngle'][frame_idx] / 180
                            obs.append(compass_scaled)

                        if 'observation_inventory' in others:
                            inventory = others['observation_inventory'][frame_idx, :] / 2304
                            obs.extend(list(inventory))

                        other_obs.append(obs)

                # (64, 64, 3) -> (3, 64, 64)
                frame = np.moveaxis(frame, -1, 0)
                frames.append(frame)
            else:
                break
            frame_idx += 1
        cap.release()

        logger.info("Number of frames: {}".format(len(frames)))

        if len(frames) < length + 1:
            raise Exception(
                "Frame length is smaller than data length.")

        # delete initial frames to align with actions
        frames = frames[-length - 1:]
        obs_q = deque([], maxlen=frame_stack)
        for i in range(frame_stack):
            if use_full_observation:
                obs_q.append((frames[0], other_obs[0]))
            else:
                obs_q.append(frames[0])

        for x in range(0, length, frameskip):
            actions = get_encoded_action(env_name, others, x, frameskip,
                                         camera_atomic_actions, max_range_of_camera)
            if x + frameskip >= length:
                done = True
            else:
                done = False

            if use_full_observation:
                obs = (LazyFrames([x[0] for x in obs_q], stack_axis=0),
                       LazyFrames([x[1] for x in obs_q], stack_axis=0))
                next_frame = frames[min(x + frameskip, length)]
                next_other = other_obs[min(x + frameskip, length)]
                obs_q.append((next_frame, next_other))
                next_obs = (LazyFrames([x[0] for x in obs_q], stack_axis=0),
                            LazyFrames([x[1] for x in obs_q], stack_axis=0))
            else:
                obs = LazyFrames(list(obs_q), stack_axis=0)
                next_frame = frames[min(x + frameskip, length)]
                obs_q.append(next_frame)
                next_obs = LazyFrames(list(obs_q), stack_axis=0)

            reward = others['reward'][x:x + frameskip].sum()

            replay_buffer.append(state=obs, action=actions, reward=reward,
                                 next_state=next_obs, next_action=None,
                                 is_state_terminal=done, demo=True)

        dr_end = time.time()
        logger.info("Time taken: {}s".format(dr_end - dr_start))

    end = time.time()
    logger.info("Total time taken: {}s".format(end - start))


def choose_top_experts(base_dir, n_experts, logger=getLogger(__name__)):
    dirs = glob.glob(os.path.join(base_dir, '*'))
    total_rewards = []

    for dr in dirs:
        data_file = os.path.join(dr, 'rendered.npz')
        data = np.load(data_file)
        total_rewards.append(data['reward'].sum())

    ids = np.argsort(total_rewards)
    chosen_dirs = []
    n_experts = min(n_experts, len(dirs))

    for i in range(n_experts):
        pos = ids[-i - 1]
        logger.info("Player: {}".format(dirs[pos]))
        logger.info("Total reward: {}".format(total_rewards[pos]))
        chosen_dirs.append(dirs[pos])

    return chosen_dirs
