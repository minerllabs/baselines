import minerl
import itertools
import gym
import sys
import numpy as np

NUM_EPISODES = 42


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


def step_env(environment='MineRLObtainTest-v0'):
    # Run random agent through environment
    env = gym.make(environment) # or try 'MineRLNavigateDense-v0'

    for _ in range(NUM_EPISODES):
        obs, info = env.reset()
        done = False

        while not done:
            # This default action has only been added for MineRLObtainIronPickaxe-v0 so far
            action = env.default_action
            # action['attack'] = 1
            action['place'] = env.action_space.spaces['place'].sample()
            # action['craft'] = env.action_space.spaces['craft'].sample()
            # action['nearbyCraft'] = env.action_space.spaces['nearbyCraft'].sample()
            # action['nearbySmelt'] = env.action_space.spaces['nearbySmelt'].sample()
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
    else:
        print("Testing data pipeline")
        step_data()
        print("Testing environment")
        step_env()
