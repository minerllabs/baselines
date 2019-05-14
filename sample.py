import minerl
import itertools
import gym
import os
import numpy as np

NUM_EPISODES = 42


def step_data():
    d = minerl.data.make('MineRLNavigate-v0')

    # Iterate through batches of data
    counter = 0
    for act, obs, rew in itertools.islice(d.batch_iter(3, None), 600):
        print("Act shape:", len(act), act)
        print("Obs shape:", len(obs), obs)
        print("Rew shape:", len(rew), rew)
        print(counter + 1)
        counter += 1


def step_env():
    # Run random agent through environment
    env = gym.make('MineRLTreechop-v0') # or try 'MineRLNavigateDense-v0' 

    for _ in range(NUM_EPISODES):
        obs, info = env.reset()
        done = False

        while not done:
            obs, reward, done, info = env.step(
                env.action_space.sample())
        print("MISSION DONE")

    print("Demo Complete.")


if __name__ == '__main__':
    # Data pipeline is not ready to be published
    print("Testing data pipeline")
    step_data()
    # print("Testing environment")
    # step_env()
