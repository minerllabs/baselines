from minerl import data
from minerl import env

import os


def main():
    d = data.init(os.path.join('C:', os.sep, 'data', 'data_texture_1_low_res'))
    # e = env.init()
    # obs = env.reset()

    for batch in d.batch_iter(64, None):
        print("Batch len:", len(batch))


    # while not env.isFinished():
    #     action = actionSpace.sample()
    #     human_obs = data.get()
    #     env_obs, reward = env.step(action)


if __name__ == '__main__':
    main()
