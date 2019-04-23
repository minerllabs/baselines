from minerl import data
from minerl import env


def main():
    # d = data.init()
    # e = env.init()
    obs = env.reset()

    while not data.finished():
        human_obs, human_reward = data.sample()

    actionSpace = env.getActionSpace()

    while not env.isFinished():
        action = actionSpace.sample()
        human_obs = data.get()
        env_obs, reward = env.step(action)

