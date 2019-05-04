import minerl
import itertools
import gym
import os

NUM_EPISODES = 42


def step_data():
    d = minerl.data.init()

    # Iterate through batches of data
    for batch in itertools.islice(d.batch_iter(64, None), 2):
        print("Batch len:", len(batch))

    # Iterate through batches of data
    for batch in itertools.islice(d.batch_iter(64, None), 3):
        print("Batch thing len:", len(batch))


    # Iterate through batches of data
    for batch in itertools.islice(d.batch_iter(64, None), 2):
        print("Batch len:", len(batch))



def step_env():
    # Run random agent through environemnt
    env = gym.make('MineRLTreechop-v0')

    for _ in range(NUM_EPISODES):
        obs, info = env.reset()
        done = False

        while not done:
            obs, reward, done, info = env.step(
                env.action_space.sample())
        print("MISSION DONE")

    print("Demo Complete.")


if __name__ == '__main__':
    step_data()
    step_env()
