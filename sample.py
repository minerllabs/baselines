import time

import minerl
import itertools
import gym
import sys
import numpy as np

NUM_EPISODES = 4


def step_data(environment='MineRLTreechop-v0'):
    d = minerl.data.make(environment)

    # Iterate through batches of data
    counter = 0
    for act, obs, rew in itertools.islice(d.batch_iter(3, None), 600):
        print("Act shape:", len(act), act)
        print("Obs shape:", len(obs), obs)
        print("Rew shape:", len(rew), rew)
        print(counter + 1)
        counter += 1


def gen_obtain_debug_actions(env):
    actions = []

    def act(**kwargs):
        action = env.default_action.copy()
        for key, value in kwargs.items():
            action[key] = value
        actions.append(action)

    act(camera=np.array([45.0, 0.0], dtype=np.float32))
    act(place='dirt')
    act(craft='stick')
    # act(craft='stick')  # Should fail - only have 1 plank here - but instead crafts more planks
    act(craft='planks')
    act(craft='crafting_table')
    # act(nearbyCraft='stone_pickaxe')  # Should fail - no crafting table in view
    act(camera=np.array([0.0, 90.0], dtype=np.float32))
    act(place='crafting_table')
    act(nearbyCraft='stone_pickaxe')
    act(camera=np.array([0.0, 90.0], dtype=np.float32))
    act(place='furnace')
    act(nearbySmelt='iron_ingot')
    act(nearbySmelt='iron_ingot')
    act(nearbySmelt='iron_ingot')

    act(camera=np.array([45.0, 0.0], dtype=np.float32))

    # Make pile to mine through (attack ground first to clear grass)
    act(attack=1)
    # for _ in range(20):
    #     act(jump=1, place='cobblestone')
    act(equip='stone_pickaxe')
    for _ in range(40):
        act(attack=1)

    return actions


def test_env(environment='MineRLObtainTest-v0'):
    env = gym.make(environment)
    env.reset()
    done = False

    for action in gen_obtain_debug_actions(env):
        obs, reward, done, info = env.step(action)
        if reward != 0:
            print(reward)

    while not done:
        obs, reward, done, info = env.step(env.default_action)
        if reward != 0:
            print(reward)
    print("MISSION DONE")



def step_env(environment='MineRLObtainIronPickaxe-v0'):
    # Run random agent through environment
    env = gym.make(environment) # or try 'MineRLNavigateDense-v0'

    for _ in range(NUM_EPISODES):
        obs, info = env.reset()
        done = False

        while not done:
            # This default action has only been added for MineRLObtainIronPickaxe-v0 so far
            action = env.default_action
            action['forward'] = env.action_space.spaces['forward'].sample()
            action['attack'] = 1
            action['place'] = env.action_space.spaces['place'].sample()
            obs, reward, done, info = env.step(action)
            if reward != 0:
                print(reward)
        print("MISSION DONE") 

    print("Demo Complete.")


if __name__ == '__main__':
    if len(sys.argv) > 0 and sys.argv[1] == 'data':
            print("Testing data pipeline")
            if len(sys.argv) > 2 and not sys.argv[2] is None:
                step_data(sys.argv[2])
            else:
                step_data()
    elif len(sys.argv) > 0 and sys.argv[1] == 'env':
            print("Testing environment")
            if len(sys.argv) > 2 and not sys.argv[2] is None:
                step_env(sys.argv[2])
            else:
                step_env()
    elif len(sys.argv) > 0 and sys.argv[1] == 'test':
        test_env()
    else:
        print("Testing data pipeline")
        step_data()
        print("Testing environment")
        step_env()
