from minerl.data import DataPipeline
from minerl import env

import os


def main():
    d = DataPipeline(os.path.join('C:', os.sep, 'data', 'data_texture_1_low_res'), 2, 32, 32)
    # e = env.init()
    # obs = env.reset()

    for batch in d.batch_iter(64, 64):
        print("Batch len:", len(batch))
        return

    # while not env.isFinished():
    #     action = actionSpace.sample()
    #     human_obs = data.get()
    #     env_obs, reward = env.step(action)


if __name__ == '__main__':
    main()
