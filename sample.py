import minerl
import itertools
import gym
import os
import numpy as np

NUM_EPISODES = 42


def step_data():
    d = minerl.data.init()

    # Iterate through batches of data
    for obs, info in itertools.islice(d.batch_iter(64, None), 6):
        print("Obs shape:", np.shape(obs))
        print("Info shape:", np.shape(info))


def step_env():
    # Run random agent through environemnt
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
    # print("Testing data pipeline")
    # step_data()
    print("Testing environment")
    step_env()
