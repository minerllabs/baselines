"""original source: https://github.com/chainer/chainerrl/pull/480

MIT License

Copyright (c) Preferred Networks, Inc.
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

from logging import getLogger

import numpy as np

import chainer
from chainer import cuda
import chainer.functions as F

from chainerrl.agents import DoubleDQN
from chainerrl.recurrent import state_kept

from chainerrl.misc.batch_states import batch_states
from chainerrl.replay_buffer import PrioritizedBuffer, PrioritizedReplayBuffer


def compute_weighted_value_loss(y, t, weights,
                                mask, clip_delta=True,
                                batch_accumulator='mean'):
    """Compute a loss for value prediction problem.

    Args:
        y (Variable or ndarray): Predicted values.
        t (Variable or ndarray): Target values.
        weights (ndarray): Weights for y, t.
        mask (ndarray): Mask to use for loss calculation
        clip_delta (bool): Use the Huber loss function if set True.
        batch_accumulator (str): 'mean' will divide loss by batchsize
    Returns:
        (Variable) scalar loss
    """
    assert batch_accumulator in ('mean', 'sum')
    y = F.reshape(y, (-1, 1))
    t = F.reshape(t, (-1, 1))
    if clip_delta:
        losses = F.huber_loss(y, t, delta=1.0)
    else:
        losses = F.square(y - t) / 2
    losses = F.reshape(losses, (-1,))
    loss_sum = F.sum(losses * weights * mask)
    if batch_accumulator == 'mean':
        loss = loss_sum / max(n_mask, 1.0)
    elif batch_accumulator == 'sum':
        loss = loss_sum
    return loss


class PrioritizedDemoReplayBuffer(PrioritizedReplayBuffer):
    """Modification of a PrioritizedReplayBuffer to have both persistent
    demonstration data and normal demonstration data.

    Args:
        capacity(int): Capacity of the buffer *excluding* expert demonstrations

    Standard PER parameters:
        alpha, beta0, betasteps, eps (float)
        normalize_by_max (bool)
    """

    def __init__(self, capacity=None,
                 alpha=0.6, beta0=0.4, betasteps=2e5, eps=0.01,
                 normalize_by_max=False, error_min=0,
                 error_max=2, num_steps=1):

        PrioritizedReplayBuffer.__init__(self, capacity=capacity,
                                         alpha=alpha, beta0=beta0,
                                         betasteps=betasteps,
                                         eps=eps,
                                         normalize_by_max=normalize_by_max,
                                         error_min=error_min,
                                         error_max=error_max,
                                         num_steps=num_steps)

        self.memory = PrioritizedBuffer(capacity)
        self.memory_demo = PrioritizedBuffer(None)

    def weights_from_probabilities(self, probabilities, min_probability):
        """Overwrite weights_from_probabilities to make beta increment explicit
        """
        if self.normalize_by_max == 'batch':
            # discard global min and compute batch min
            min_probability = np.min(min_probability)
        if self.normalize_by_max:
            weights = [(p / min_probability) ** -self.beta
                       for p in probabilities]
        else:
            memory_length = (len(self.memory) + len(self.memory_demo))
            weights = [(memory_length * p) ** -self.beta
                       for p in probabilities]
        return weights

    def update_beta(self):
        # Update beta towards 1.
        self.beta = min(1.0, self.beta + self.beta_add)

    def sample_from_memory(self, nsample_agent, nsample_demo):
        """Samples experiences from memory

        Args:
            nsample_agent (int): Number of RL transitions to sample
            nsample_demo (int): Number of demonstration transitions to sample
        """
        if nsample_demo > 0:
            sampled_demo, prob_demo, min_prob_demo = self.memory_demo.sample(
                nsample_demo)
        else:
            sampled_demo, prob_demo, min_prob_demo = [], [], 1e+10

        if nsample_agent > 0:
            sampled_agent, prob_agent, min_prob_agent = self.memory.sample(
                nsample_agent)
        else:
            sampled_agent, prob_agent, min_prob_agent = [], [], 1e+10

        min_prob = min(min_prob_demo, min_prob_agent)

        if nsample_demo > 0:
            weights_demo = self.weights_from_probabilities(prob_demo, min_prob)
            for e, w in zip(sampled_demo, weights_demo):
                e[0]['weight'] = w

        if nsample_agent > 0:
            weights_agent = self.weights_from_probabilities(
                prob_agent, min_prob)
            for e, w in zip(sampled_agent, weights_agent):
                e[0]['weight'] = w

        return sampled_agent, sampled_demo

    def sample(self, n, demo_only=False):
        """Sample `n` experiences from memory.

        Args:
            n (int): Number of experiences to sample
            demo_only (bool): Force all samples to be drawn from demo buffer
        """
        if demo_only:
            _, sampled_demo = self.sample_from_memory(nsample_agent=0,
                                                      nsample_demo=n)
            return sampled_demo

        psum_agent = self.memory.priority_sums.sum()
        psum_demo = self.memory_demo.priority_sums.sum()
        psample_agent = psum_agent / (psum_agent + psum_demo)

        nsample_agent = np.random.binomial(n, psample_agent)
        # If we don't have enough RL transitions yet, force more demos
        nsample_agent = min(nsample_agent, len(self.memory))
        nsample_demo = n - nsample_agent

        sampled_agent, sampled_demo = self.sample_from_memory(
            nsample_agent, nsample_demo)

        return sampled_agent, sampled_demo

    def update_errors(self, errors_agent, errors_demo):
        if len(errors_demo) > 0:
            self.memory_demo.set_last_priority(
                self.priority_from_errors(errors_demo))
        if len(errors_agent) > 0:
            self.memory.set_last_priority(
                self.priority_from_errors(errors_agent))

    def append(self, state, action, reward, next_state=None, next_action=None,
               is_state_terminal=False, env_id=0, demo=False, **kwargs):
        """
        Args:
            demo: Flags transition as a demonstration and store it persistently
        """
        memory = self.memory_demo if demo else self.memory
        last_n_transitions = self.last_n_transitions[env_id]
        experience = dict(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            next_action=next_action,
            is_state_terminal=is_state_terminal,
            **kwargs
        )
        memory.append([experience])
        last_n_transitions.append(experience)
        if is_state_terminal:
            while last_n_transitions:
                memory.append(list(last_n_transitions))
                del last_n_transitions[0]
            assert len(last_n_transitions) == 0
        else:
            if len(last_n_transitions) == self.num_steps:
                memory.append(list(last_n_transitions))

    def stop_current_episode(self, demo=False, env_id=0):
        memory = self.memory_demo if demo else self.memory
        last_n_transitions = self.last_n_transitions[env_id]
        # if n-step transition hist is not full, add transition;
        # if n-step hist is indeed full, transition has already been added;
        if 0 < len(last_n_transitions) < self.num_steps:
            memory.append(list(last_n_transitions))
        # avoid duplicate entry
        if 0 < len(last_n_transitions) <= self.num_steps:
            del last_n_transitions[0]
        while last_n_transitions:
            memory.append(list(last_n_transitions))
            del last_n_transitions[0]
        assert len(last_n_transitions) == 0

    def __len__(self):
        return len(self.memory) + len(self.memory_demo)


class DemoReplayUpdater(object):
    """Object that handles update schedule and configurations.

    Args:
        replay_buffer (PrioritizedDemoReplayBuffer): Bbuffer for self-play
        update_func (callable): Callable that accepts one of these:
            (1) two lists of transition dicts (if episodic_update=False)
            (2) two lists of transition dicts (if episodic_update=True)
        batchsize (int): Minibatch size
        update_interval (int): Model update interval in step
        n_times_update (int): Number of repetition of update
        episodic_update (bool): Use full episodes for update if set True
        episodic_update_len (int or None): Subsequences of this length are used
            for update if set int and episodic_update=True
    """

    def __init__(self, replay_buffer,
                 update_func, batchsize, episodic_update,
                 n_times_update, replay_start_size, update_interval,
                 episodic_update_len=None):
        assert batchsize <= replay_start_size
        self.replay_buffer = replay_buffer
        self.update_func = update_func
        self.batchsize = batchsize
        self.episodic_update = episodic_update
        self.episodic_update_len = episodic_update_len
        self.n_times_update = n_times_update
        self.replay_start_size = replay_start_size
        self.update_interval = update_interval

    def update_if_necessary(self, iteration):
        """Called during normal self-play
        """
        if len(self.replay_buffer) < self.replay_start_size:
            return

        if (self.episodic_update and (
                self.replay_buffer.n_episodes < self.batchsize)):
            return

        if iteration % self.update_interval != 0:
            return

        for _ in range(self.n_times_update):
            if self.episodic_update:
                raise NotImplementedError()
            else:
                transitions_agent, transitions_demo = self.replay_buffer.sample(
                    self.batchsize)
                self.update_func(transitions_agent, transitions_demo)
                # Update beta only during RL
                self.replay_buffer.update_beta()

    def update_from_demonstrations(self):
        """Called during pre-train steps. All samples are from demo buffer
        """
        if self.episodic_update:
            episodes_demo = self.replay_buffer.sample_episodes(
                self.batch_size, self.episodic_update_len)
            self.update_func([], episodes_demo)
        else:
            transitions_demo = self.replay_buffer.sample(
                self.batchsize, demo_only=True)
            self.update_func([], transitions_demo)


def batch_experiences(experiences, xp, phi, reward_transform, gamma,
                      batch_states=batch_states):
    """Takes a batch of k experiences each of which contains j
    consecutive transitions and vectorizes them, where j is between 1 and n.
    Args:
        experiences: list of experiences. Each experience is a list
            containing between 1 and n dicts containing
              - state (object): State
              - action (object): Action
              - reward (float): Reward
              - is_state_terminal (bool): True iff next state is terminal
              - next_state (object): Next state
        xp : Numpy compatible matrix library: e.g. Numpy or CuPy.
        phi : Preprocessing function
        gamma: discount factor
        batch_states: function that converts a list to a batch
    Returns:
        dict of batched transitions

    Changes from chainerrl.replay_buffer.batch_experiences:
        Calculates and stores both n_step and 1_step reward
    """
    batch_exp = {
        'state': batch_states(
            [elem[0]['state'] for elem in experiences], xp, phi),
        'action': xp.asarray([elem[0]['action'] for elem in experiences]),
        'reward': xp.asarray([sum((gamma ** i) * reward_transform(exp[i]['reward'])
                                  for i in range(len(exp)))
                              for exp in experiences],
                             dtype=xp.float32),
        'next_state': batch_states(
            [elem[-1]['next_state']
             for elem in experiences], xp, phi),
        'is_n_step': xp.asarray([float(len(elem) > 1) for elem in experiences],
                                dtype=xp.float32),
        'is_state_terminal': xp.asarray(
            [any(transition['is_state_terminal']
                 for transition in exp) for exp in experiences],
            dtype=xp.float32),
        'discount': xp.asarray([(gamma ** len(elem))for elem in experiences],
                               dtype=xp.float32)}
    if all(elem[-1]['next_action'] is not None for elem in experiences):
        batch_exp['next_action'] = xp.asarray(
            [elem[-1]['next_action'] for elem in experiences])
    return batch_exp


class DQfD(DoubleDQN):
    """Deep-Q Learning from Demonstrations
    See: https://arxiv.org/abs/1704.03732.

    DQN Args:
        q_function (StateQFunction): Q-function
        optimizer (Optimizer): Optimizer that is already setup
        replay_buffer (PrioritizedDemoReplayBuffer): Replay buffer
        gamma (float): Discount factor
        explorer (Explorer): Explorer that specifies an exploration strategy.
        gpu (int): GPU device id if not None nor negative.
        replay_start_size (int): if the replay buffer's size is less than
            replay_start_size, skip update
        minibatch_size (int): Minibatch size
        update_interval (int): Model update interval in step
        target_update_interval (int): Target model update interval in step
        clip_delta (bool): Clip delta if set True
        phi (callable): Feature extractor applied to observations
        target_update_method (str): 'hard' or 'soft'.
        soft_update_tau (float): Tau of soft target update.
        n_times_update (int): Number of repetition of update
        average_q_decay (float): Decay rate of average Q, only used for
            recording statistics
        average_loss_decay (float): Decay rate of average loss, only used for
            recording statistics
        batch_accumulator (str): 'mean' or 'sum'
        episodic_update (bool): Use full episodes for update if set True
        episodic_update_len (int or None): Subsequences of this length are used
            for update if set int and episodic_update=True
        logger (Logger): Logger used
        batch_states (callable): method which makes a batch of observations.
            default is `chainerrl.misc.batch_states.batch_states`

    DQfD-specific args:
        n_pretrain_steps: Number of pretraining steps to perform
        demo_supervised_margin (float): Margin width for supervised demo loss
        loss_coeff_nstep(float): Coefficient used to regulate n-step q loss
        loss_coeff_supervised (float): Coefficient for the supervised loss term
        loss_coeff_l2 (float): Coefficient used to regulate weight decay rate
        bonus_priority_agent(float): Bonus priorities for agent generated data
        bonus_priority_demo (float): Bonus priorities for demonstration data
        reward_transform (callable): Function that changes the scale of the reward
    """

    def __init__(self, q_function, optimizer,
                 replay_buffer,
                 gamma, explorer, n_pretrain_steps,
                 demo_supervised_margin=0.8,
                 bonus_priority_agent=0.001,
                 bonus_priority_demo=1.0,
                 loss_coeff_nstep=1.0,
                 loss_coeff_supervised=1.0,
                 loss_coeff_l2=1e-5, gpu=None,
                 replay_start_size=50000,
                 minibatch_size=32, update_interval=1,
                 target_update_interval=10000, clip_delta=True,
                 phi=lambda x: x,
                 reward_transform=lambda x: x,
                 target_update_method='hard',
                 soft_update_tau=1e-2,
                 n_times_update=1, average_q_decay=0.999,
                 average_loss_decay=0.99,
                 batch_accumulator='mean', episodic_update=False,
                 episodic_update_len=None,
                 logger=getLogger(__name__),
                 batch_states=batch_states):

        assert isinstance(replay_buffer, PrioritizedDemoReplayBuffer)
        super(DQfD, self).__init__(q_function, optimizer, replay_buffer, gamma,
                                   explorer, gpu, replay_start_size,
                                   minibatch_size, update_interval,
                                   target_update_interval, clip_delta,
                                   phi, target_update_method, soft_update_tau,
                                   n_times_update, average_q_decay,
                                   average_loss_decay, batch_accumulator,
                                   episodic_update, episodic_update_len,
                                   logger, batch_states)

        self.minibatch_size = minibatch_size
        self.n_pretrain_steps = n_pretrain_steps
        self.demo_supervised_margin = demo_supervised_margin
        self.loss_coeff_supervised = loss_coeff_supervised
        self.loss_coeff_l2 = loss_coeff_l2
        self.loss_coeff_nstep = loss_coeff_nstep
        self.bonus_priority_demo = bonus_priority_demo
        self.bonus_priority_agent = bonus_priority_agent
        self.reward_transform = reward_transform

        self.average_demo_td_error = 0
        self.average_agent_td_error = 0

        self.average_loss_1step = 0
        self.average_loss_nstep = 0
        self.average_loss_supervised = 0
        self.average_iweights = 0

        self.optimizer.add_hook(
            chainer.optimizer_hooks.WeightDecay(loss_coeff_l2))

        # Overwrite DQN's replay updater.
        self.replay_updater = DemoReplayUpdater(
            replay_buffer=self.replay_buffer,
            update_func=self.update,
            batchsize=minibatch_size,
            episodic_update=episodic_update,
            episodic_update_len=episodic_update_len,
            n_times_update=n_times_update,
            replay_start_size=replay_start_size,
            update_interval=update_interval,
        )

    def pretrain(self):
        """Uses purely expert demonstrations to do pre-training
        """
        logger = getLogger(__name__)
        for tpre in range(self.n_pretrain_steps):
            self.replay_updater.update_from_demonstrations()
            if tpre % self.target_update_interval == 0:
                logger.info('PRETRAIN-step:%s statistics:%s',
                            tpre, self.get_statistics())
                self.sync_target_network()

    def update(self, experiences_agent, experiences_demo):
        """Combined DQfD loss function for Demonstration and agent/RL.
        """
        num_exp_agent = len(experiences_agent)
        experiences = experiences_agent+experiences_demo
        exp_batch = batch_experiences(experiences, xp=self.xp, phi=self.phi,
                                      reward_transform=self.reward_transform,
                                      gamma=self.gamma,
                                      batch_states=self.batch_states)

        exp_batch['weights'] = self.xp.asarray(
            [elem[0]['weight']for elem in experiences], dtype=self.xp.float32)

        errors_out = []
        loss_q_nstep, loss_q_1step = self._compute_ddqn_losses(
            exp_batch, errors_out=errors_out)

        # Add the agent/demonstration bonus priorities and update
        err_agent = errors_out[:num_exp_agent]
        err_demo = errors_out[num_exp_agent:]
        err_agent = [e+self.bonus_priority_agent for e in err_agent]
        err_demo = [e+self.bonus_priority_demo for e in err_demo]
        self.replay_buffer.update_errors(err_agent, err_demo)

        # Large-margin supervised loss
        # Grab the cached Q(s) in the forward pass & subset demo exp.
        q_picked = self.qout.evaluate_actions(exp_batch["action"])
        q_expert_demos = q_picked[num_exp_agent:]

        # unwrap DiscreteActionValue and subset demos
        q_demos = []

        for branch in self.qout.branches:
            q_demos.append(branch.q_values[num_exp_agent:])

        q_demos = F.hstack(q_demos)

        # Calculate margin forall actions (l(a_E,a) in the paper)
        margin = self.xp.zeros_like(
            q_demos.array) + self.demo_supervised_margin
        a_expert_demos = exp_batch["action"][num_exp_agent:]
        pos = 0
        for i, branch in enumerate(self.qout.branches):
            margin[self.xp.arange(len(experiences_demo)),
                   a_expert_demos[:, i] + pos] = 0
            pos += branch.q_values.shape[1]

        # Supervised targets
        q_demos_penalized = q_demos + margin
        supervised_targets = []
        pos = 0
        for i, branch in enumerate(self.qout.branches):
            branch_size = branch.q_values.shape[1]
            supervised_targets.append(
                F.max(q_demos_penalized[:, pos:pos + branch_size], axis=-1).reshape(-1, 1))
            pos += branch_size
        supervised_targets = F.hstack(supervised_targets)

        # Supervised loss calculation
        n_branches = len(self.qout.branches)
        iweights_demos = exp_batch['weights'][num_exp_agent:].reshape(-1, 1)
        iweights_demos = self.xp.tile(iweights_demos, (1, n_branches))
        loss_supervised = F.squared_error(supervised_targets, q_expert_demos)
        loss_supervised = F.sum(iweights_demos * loss_supervised)
        if self.batch_accumulator == "mean":
            loss_supervised /= n_branches
            loss_supervised /= max(len(experiences_demo), 1)

        loss_combined = loss_q_1step + \
            self.loss_coeff_nstep * loss_q_nstep + \
            self.loss_coeff_supervised * loss_supervised

        self.model.cleargrads()
        loss_combined.backward()
        self.optimizer.update()

        self.average_loss *= self.average_loss_decay
        self.average_loss += (1 - self.average_loss_decay) * \
            float(loss_combined.array)

        self.average_loss_1step *= self.average_loss_decay
        self.average_loss_1step += (1 - self.average_loss_decay) * \
            float(loss_q_1step.array)

        self.average_loss_nstep *= self.average_loss_decay
        self.average_loss_nstep += (1 - self.average_loss_decay) * \
            float(loss_q_nstep.array)

        self.average_loss_supervised *= self.average_loss_decay
        self.average_loss_supervised += (1 - self.average_loss_decay) * \
            float(loss_supervised.array)

        self.average_iweights *= self.average_loss_decay
        self.average_iweights += (1 - self.average_loss_decay) * \
            cuda.to_cpu(exp_batch['weights'].mean())

        if len(err_demo):
            self.average_demo_td_error *= self.average_loss_decay
            self.average_demo_td_error += (1 - self.average_loss_decay) * \
                np.mean(err_demo)

        if len(err_agent):
            self.average_agent_td_error *= self.average_loss_decay
            self.average_agent_td_error += (1 - self.average_loss_decay) * \
                np.mean(err_agent)

    def _compute_target_values(self, exp_batch):
        batch_next_state = exp_batch['next_state']

        with chainer.using_config('train', False), state_kept(self.q_function):
            next_qout = self.q_function(batch_next_state)

        target_next_qout = self.target_q_function(batch_next_state)

        next_q_max = target_next_qout.evaluate_actions(
            next_qout.greedy_actions)
        next_q_max = F.mean(next_q_max, axis=1)

        batch_rewards = exp_batch['reward']
        batch_terminal = exp_batch['is_state_terminal']
        discount = exp_batch['discount']

        return batch_rewards + discount * (1.0 - batch_terminal) * next_q_max

    def _compute_y_and_ts(self, exp_batch):
        """Compute output and targets

        Changes from DQN:
            Cache qout for the supervised loss later
            Calculate both 1-step and n-step targets
        """
        # Compute Q-values for current states
        batch_state = exp_batch['state']
        qout = self.model(batch_state)

        # Caches Q(s) for use in supervised demo loss
        self.qout = qout

        batch_actions = exp_batch['action']
        batch_q = qout.evaluate_actions(batch_actions)

        with chainer.no_backprop_mode():
            # Calculate Double DQN targets
            batch_q_target = self._compute_target_values(
                exp_batch).reshape((-1, 1))
            batch_q_target = F.broadcast_to(batch_q_target, batch_q.shape)
        return batch_q, batch_q_target

    def _compute_ddqn_losses(self, exp_batch, errors_out=None):
        """Compute the Q-learning losses for a batch of experiences

        Args:
          exp_batch (dict): A dict of batched arrays of transitions
        Returns:
          Computed loss from the minibatch of experiences
        """
        y, t = self._compute_y_and_ts(exp_batch)
        n_branches = exp_batch['action'].shape[1]

        # Calculate the errors_out for priorities with the 1-step err
        del errors_out[:]
        delta = F.absolute(y - t)
        if delta.ndim == 2:
            delta = F.sum(delta, axis=1)
        delta = cuda.to_cpu(delta.array)
        for e in delta:
            errors_out.append(e)

        is_1_step = self.xp.abs(1. - exp_batch["is_n_step"]).reshape(-1, 1)
        is_1_step = self.xp.tile(is_1_step, (1, n_branches)).reshape(-1)
        is_n_step = exp_batch['is_n_step'].reshape(-1, 1)
        is_n_step = self.xp.tile(is_n_step, (1, n_branches)).reshape(-1)
        weights = exp_batch['weights'].reshape(-1, 1)
        weights = F.tile(weights, (1, n_branches)).reshape(-1)
        loss_1step = compute_weighted_value_loss(
            y, t, weights,
            mask=is_1_step,
            clip_delta=self.clip_delta,
            batch_accumulator=self.batch_accumulator)
        loss_nstep = compute_weighted_value_loss(
            y, t, weights,
            mask=is_n_step,
            clip_delta=self.clip_delta,
            batch_accumulator=self.batch_accumulator)

        return loss_nstep, loss_1step

    def act(self, obs):
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            action_value = self.model(
                self.batch_states([obs], self.xp, self.phi))
            q = cuda.to_cpu(action_value.max.array.reshape(-1))
            action = cuda.to_cpu(action_value.greedy_actions.array).reshape(-1)

        self.logger.debug('t:%s q:%s action_value:%s', self.t, q, action_value)
        return action

    def act_and_train(self, obs, reward):

        with chainer.using_config('train', False), chainer.no_backprop_mode():
            action_value = self.model(
                self.batch_states([obs], self.xp, self.phi))
            q = cuda.to_cpu(action_value.max.array.reshape(-1))
            greedy_action = cuda.to_cpu(
                action_value.greedy_actions.array).reshape(-1)

        self.logger.debug('t:%s q:%s action_value:%s', self.t, q, action_value)

        action = self.explorer.select_action(
            self.t, lambda: greedy_action, action_value=action_value)
        self.t += 1

        # Update the target network
        if self.t % self.target_update_interval == 0:
            self.sync_target_network()

        if self.last_state is not None:
            assert self.last_action is not None
            # Add a transition to the replay buffer
            self.replay_buffer.append(
                state=self.last_state,
                action=self.last_action,
                reward=reward,
                next_state=obs,
                next_action=action,
                is_state_terminal=False)

        self.last_state = obs
        self.last_action = action

        self.replay_updater.update_if_necessary(self.t)
        self.logger.debug('t:%s r:%s a:%s', self.t, reward, action)

        return self.last_action

    def get_statistics(self):
        return [
            ('average_loss_1step', self.average_loss_1step),
            ('average_loss_nstep', self.average_loss_nstep),
            ('average_loss_supervised', self.average_loss_supervised),
            ('average_iweights', self.average_iweights),
            ('average_loss', self.average_loss),
            ('n_updates', self.optimizer.t),
        ]
