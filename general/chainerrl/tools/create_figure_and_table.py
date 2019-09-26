ALGO_PATHS = {
    'Rainbow': [
        '/mnt/vol12/nakata/home/minerl2019/baselines/general/chainerrl/baselines/results/v23_20190906/MineRLTreechop-v0/rainbow/20190906T122521.545748/log',
        '/mnt/vol12/nakata/home/minerl2019/baselines/general/chainerrl/baselines/results/v23_20190906/MineRLTreechop-v0/rainbow/20190906T122521.739546/log',
        '/mnt/vol12/nakata/home/minerl2019/baselines/general/chainerrl/baselines/results/v23_20190906/MineRLTreechop-v0/rainbow/20190906T122522.864753/log',
    ],
    'PPO': [
        '/mnt/vol12/nakata/home/minerl2019/baselines/general/chainerrl/baselines/results/v23_20190906/MineRLTreechop-v0/ppo/20190906T122914.023884/log',
        '/mnt/vol12/nakata/home/minerl2019/baselines/general/chainerrl/baselines/results/v23_20190906/MineRLTreechop-v0/ppo/20190906T122530.781642/log',
        '/mnt/vol12/nakata/home/minerl2019/baselines/general/chainerrl/baselines/results/v23_20190906/MineRLTreechop-v0/ppo/20190906T122531.377449/log',
    ],
    'DDDQN': [
        '/mnt/vol12/nakata/home/minerl2019/baselines/general/chainerrl/baselines/results/v23_20190906/MineRLTreechop-v0/dddqn/20190906T122513.099442/log',
        '/mnt/vol12/nakata/home/minerl2019/baselines/general/chainerrl/baselines/results/v23_20190906/MineRLTreechop-v0/dddqn/20190906T122511.222353/log',
        '/mnt/vol12/nakata/home/minerl2019/baselines/general/chainerrl/baselines/results/v23_20190906/MineRLTreechop-v0/dddqn/20190906T122511.901882/log',
    ],
}
FIGURE_TITLE = 'MineRLTreechop-v0'
FIGURE_OUT = '/mnt/vol12/nakata/home/minerl2019/baselines/general/chainerrl/tools/hoge.png'
BEST_SCORE_OUT = '/mnt/vol12/nakata/home/minerl2019/baselines/general/chainerrl/tools/hoge.txt'

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def read_data(algo_paths):
    algo_dfs = {}
    for algo, log_paths in algo_paths.items():
        dfs = []
        for log_path in log_paths:
            dfs.append(pd.read_json(log_path))
        algo_dfs[algo] = dfs
    return algo_dfs


def print_best_score_with_std(algo_dfs):
    def _get_best_score_with_std_of_an_algo(dfs):
        # find the best 100 contiguous episodes among trials
        best_score = -np.inf
        best_std = None

        for df in dfs:
            reward_data = df['training/reward']
            for epi in range(len(reward_data) - 100):
                target_data = reward_data[epi:epi + 100]
                score = np.nanmean(target_data)
                if score > best_score:
                    best_score = score
                    best_std = np.nanstd(target_data)
        return (best_score, best_std)

    algo_best_scores = {}
    for algo_name, dfs in algo_dfs.items():
        best_score, best_std = _get_best_score_with_std_of_an_algo(dfs)
        algo_best_scores[algo_name] = (best_score, best_std)
    return algo_best_scores


def create_figure(algo_dfs, fig):
    def _plot_an_algo(label, dfs, ax):
        maxepi = max(len(df) for df in dfs)
        n_trial = len(dfs)

        reward_data = np.full((maxepi, n_trial), np.nan)

        for trial, df in enumerate(dfs):
            for epi, reward in enumerate(df['training/reward']):
                reward_data[epi, trial] = reward

        avg_data = []
        std_data = []
        for epi in range(reward_data.shape[0]):
            start = epi - 15
            if start < 0:
                start = 0
            end = epi + 15

            target_data = reward_data[start:end, :].ravel()
            avg_data.append(np.nanmean(target_data))
            std_data.append(np.nanstd(target_data))
        avg_data = np.asarray(avg_data)
        std_data = np.asarray(std_data)

        x = np.arange(maxepi)
        ax.plot(x, avg_data, label=label, alpha=0.5)
        ax.fill_between(x, avg_data - std_data, avg_data + std_data, alpha=0.2)

    ax = fig.add_subplot(111)

    for algo_name, dfs in algo_dfs.items():
        _plot_an_algo(algo_name, dfs, ax)

    # legend/labels
    ax.legend(loc='best')
    ax.set_xlabel('episode')
    ax.set_ylabel('training/reward (average over 30 episodes)')
    ax.set_title(FIGURE_TITLE)
    ax.grid(True)

    return fig


def tee(msg, fname, mode='a'):
    print(msg)
    with open(fname, mode=mode) as f:
        print(msg, file=f)


def main():
    algo_dfs = read_data(ALGO_PATHS)

    # plot
    plt.rcParams['font.size'] = 15
    fig = plt.figure(figsize=(12, 8))
    fig = create_figure(algo_dfs, fig)
    fig.savefig(FIGURE_OUT)
    print('figure saved at {}'.format(FIGURE_OUT))
    plt.close(fig)

    # best scores (for table)
    algo_best_scores = print_best_score_with_std(algo_dfs)
    for algo_name, (best_score, best_std) in algo_best_scores.items():
        msg = '{}: best_score: {} +- {} ("+-" denotes standard deviation)'.format(algo_name, best_score, best_std)
        tee(msg, BEST_SCORE_OUT, mode='a')
    print('best score saved at {}'.format(BEST_SCORE_OUT))


if __name__ == '__main__':
    main()
