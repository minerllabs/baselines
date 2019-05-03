import minerl
import gym

import os


def step_data():
    d = minerl.data.init(os.path.join('C:', os.sep, 'data', 'data_texture_1_low_res'))

    # Iterate through batches of data
    for batch in d.batch_iter(64, None):
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
