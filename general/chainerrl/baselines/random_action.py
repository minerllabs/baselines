import argparse
from logging import getLogger
import os

import minerl  # noqa: register MineRL envs as Gym envs.
import gym

import chainerrl

import sys
sys.path.append(os.path.abspath(os.path.join(__file__, os.pardir)))
import utils
from env_wrappers import ResetTrimInfoWrapper, ObtainPoVWrapper, ContinuingTimeLimitMonitor

logger = getLogger(__name__)


class StaticPolicyAgent(chainerrl.agent.AttributeSavingMixin, chainerrl.agent.Agent):
    saved_attributes = ()

    def __init__(self, action_sampler, logger=getLogger(__name__)):
        self._action_sampler = action_sampler
        self._logger = logger

    def act_and_train(self, obs, reward):
        raise NotImplementedError('StaticPolicyAgent is never trained.')

    def act(self, obs):
        return self._action_sampler()

    def stop_episode_and_train(self, state, reward, done=False):
        pass

    def stop_episode(self):
        pass

    def get_statistics(self):
        raise NotImplementedError('StaticPolicyAgent is never trained.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='MineRLTreechop-v0',
                        choices=[
                            'MineRLTreechop-v0',
                            'MineRLNavigate-v0', 'MineRLNavigateDense-v0',
                            'MineRLNavigateExtreme-v0', 'MineRLNavigateExtremeDense-v0',
                            'MineRLObtainIronPickaxe-v0', 'MineRLObtainIronPickaxeDense-v0',
                            'MineRLObtainDiamond-v0', 'MineRLObtainDiamondDense-v0',
                            # for debug use
                            'MineRLNavigateDenseFixed-v0',
                            'MineRLObtainTest-v0',
                        ],
                        help='MineRL environment identifier.')
    parser.add_argument('--outdir', type=str, default='results',
                        help='Directory path to save output files. If it does not exist, it will be created.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed [0, 2 ** 31)')
    parser.add_argument('--eval-n-runs', type=int, default=10)
    parser.add_argument('--logging-level', type=int, default=20, help='Logging level. 10:DEBUG, 20:INFO etc.')
    parser.add_argument('--monitor', action='store_true', default=False,
                        help='Monitor env. Videos and additional information are saved as output files when evaluation.')
    args = parser.parse_args()

    args.outdir = chainerrl.experiments.prepare_output_dir(args, args.outdir)

    import logging
    log_format = '%(levelname)-8s - %(asctime)s - [%(name)s %(funcName)s %(lineno)d] %(message)s'
    logging.basicConfig(filename=os.path.join(args.outdir, 'log.txt'), format=log_format, level=args.logging_level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(args.logging_level)
    console_handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger('').addHandler(console_handler)  # add hander to the root logger

    logger.info('Output files are saved in {}'.format(args.outdir))

    utils.log_versions()

    try:
        _main(args)
    except:  # noqa
        logger.exception('execution failed.')
        raise


def _main(args):
    logger.info('The first `gym.make(MineRL*)` may take several minutes. Be patient!')

    os.environ['MALMO_MINECRAFT_OUTPUT_LOGDIR'] = args.outdir

    # Set a random seed used in ChainerRL.
    chainerrl.misc.set_random_seed(args.seed)

    # Set different random seeds for train and test envs.
    train_seed = args.seed  # noqa: never used in this script
    test_seed = 2 ** 31 - 1 - args.seed

    def wrap_env(env, test):
        if test and args.monitor:
            # NOTE: wrapping order matters!
            env = ObtainPoVWrapper(env)
            env = ContinuingTimeLimitMonitor(
                env, os.path.join(args.outdir, 'monitor'),
                mode='evaluation' if test else 'training', video_callable=lambda episode_id: True)
        env_seed = test_seed if test else train_seed
        # env.seed(int(env_seed))  # TODO: not supported yet
        return env

    core_env = gym.make(args.env)
    eval_env = wrap_env(core_env, test=True)

    # build agent
    action_sampler = eval_env.action_space.sample
    agent = StaticPolicyAgent(action_sampler)

    # experiment
    eval_stats = chainerrl.experiments.eval_performance(env=eval_env, agent=agent, n_steps=None, n_episodes=args.eval_n_runs)
    eval_stats_txt = 'n_episodes: {} mean: {} median: {} stdev {}'.format(
        eval_stats['episodes'], eval_stats['mean'], eval_stats['median'], eval_stats['stdev'])
    logger.info(eval_stats_txt)
    with open(os.path.join(args.outdir, 'scores.txt'), 'w') as f:
        print(eval_stats_txt, file=f)

    eval_env.close()


if __name__ == '__main__':
    main()
