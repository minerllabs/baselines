"""convert ChainerRL's `log.txt` into Chainer's standard style `log` json file"""

import argparse
import os
import re
import json
import shutil
import logging

import numpy as np

logger = logging.getLogger(__name__)


pattern_epi = re.compile(r'^.*step:(\d+) episode:(\d+) R:(-?[0-9.]+)$')
pattern_stat = re.compile(r'^.*statistics:(.*)$')
pattern_eval = re.compile(r'^.*evaluation episode (\d+) length:(\d+) R:(-?[0-9.]+)$')


def parse_line_episode(line):
    """
    convert a line:
    INFO     - 2019-06-13 11:30:56,053 - [chainerrl.experiments.train_agent train_agent 67] outdir:results/MineRLTreechop-v0/rainbow_frameskip32_stack4/20190613T102910.020709 step:7459 episode:32 R:5.0
    into:
    `(7459, 32, 5.0)`
    """
    m = re.match(pattern_epi, line)
    step = int(m.group(1))
    episode = int(m.group(2))
    reward = float(m.group(3))
    return step, episode, reward


def parse_line_stat(line):
    """
    convert a line like:
    INFO     - 2019-06-06 10:57:50,747 - [chainerrl.experiments.train_agent train_agent 68] statistics:[('average_q', -0.059951474330555914), ('average_loss', 0.0037793767816309155), ('n_updates', 1000)]
    into:
    `{'average_q': -0.059951474330555914, 'average_loss': 0.0037793767816309155, 'n_updates': 1000}`
    """
    m = re.match(pattern_stat, line)
    s = m.group(1).replace('nan)', 'None)')
    d = dict(eval(s))
    return d


def parse_line_evaluation(line):
    """
    convert a line:
    INFO     - 2019-06-21 00:29:17,070 - [chainerrl.experiments.train_agent run_evaluation_episodes 76] evaluation episode 0 length:500 R:0.0
    into:
    `(0, 500, 0.0)`
    """
    m = re.match(pattern_eval, line)
    eval_episode = int(m.group(1))
    length = int(m.group(2))
    reward = float(m.group(3))
    return eval_episode, length, reward


def convert_log(filepath):
    assert filepath is not None
    assert os.path.exists(filepath)

    outdir = os.path.dirname(filepath)

    scores = []
    with open(filepath) as f:
        eval_episodes, eval_lengths, eval_rewards = [], [], []
        for line_num, line in enumerate(f, start=1):
            try:
                if 'evaluation episode' in line:
                    eval_episode, length, reward = parse_line_evaluation(line)
                    eval_episodes.append(eval_episode)
                    eval_lengths.append(length)
                    eval_rewards.append(reward)
                elif 'outdir:' in line:
                    if len(eval_episodes) > 0:  # append last evaluation score to the last entry
                        entry = scores[-1]
                        entry['evaluation/n_episodes'] = len(eval_episodes)
                        entry['evaluation/length/mean'] = float(np.mean(eval_lengths))
                        entry['evaluation/length/max'] = float(np.max(eval_lengths))
                        entry['evaluation/length/min'] = float(np.min(eval_lengths))
                        entry['evaluation/length/median'] = float(np.median(eval_lengths))
                        entry['evaluation/reward/mean'] = float(np.mean(eval_rewards))
                        entry['evaluation/reward/max'] = float(np.max(eval_rewards))
                        entry['evaluation/reward/min'] = float(np.min(eval_rewards))
                        entry['evaluation/reward/median'] = float(np.median(eval_rewards))
                        scores[-1] = entry
                        eval_episodes, eval_lengths, eval_rewards = [], [], []

                    entry = {}
                    step, episode, reward = parse_line_episode(line)
                    entry['step'] = step
                    entry['episode'] = episode
                    entry['training/reward'] = reward

                    scores.append(entry)
                elif 'statistics:' in line:
                    parsed_dict = parse_line_stat(line)
                    entry = scores[-1]  # append last evaluation score to the last entry
                    entry.update(parsed_dict)
                    scores[-1] = entry
            except Exception:
                logger.exception(f'{filepath}: failed at line {line_num}')
                raise

    out = os.path.join(outdir, 'log')
    with open(out, 'w') as f:
        json.dump(scores, f, indent=2)
    print('Dump `log` as {}'.format(out))

    args_in = os.path.join(outdir, 'args.txt')
    if os.path.exists(args_in):
        shutil.copyfile(args_in, (os.path.join(outdir, 'args')))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default=None, help='Path to `log.txt`')
    args = parser.parse_args()
    convert_log(args.file)


if __name__ == '__main__':
    main()
