"""
MIT License

Copyright (c) Preferred Networks, Inc.
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import numpy as np
import chainer
from chainer import cuda
import chainer.functions as F
from chainerrl.agent import AttributeSavingMixin, BatchAgent


class Discriminator(AttributeSavingMixin):
    """ Discriminator for Generative Adversarial Imination Learning
    See https://arxiv.org/abs/1606.03476
    Args:
        model (chainer.Chain): Model
        optimizer (chainer.Optimizer): Optimizer to train model
        obs_normalizer (chainerrl.links.EmpiricalNormalization or None):
            If set to chainerrl.links.EmpiricalNormalization, it is used to
            normalize observations based on the empirical mean and standard
            deviation of observations. These statistics are updated after
            computing advantages and target values and before updating the
            policy and the value function.
        update_interval (int): Interval steps of discriminator iterations.
            Every after this amount of steps, this agent updates
            the discriminator using data from these steps and experts.
        minibatch_size (int): Minibatch size
        epochs (int): Training epochs in an update
        entropy_coef (float): Weight coefficient of discriminator for entropy
            bonus [0, inf)
        loss_decay (float): Decay rate of average loss of
            discriminator, only used for recording statistics
        entropy_decay (float): Decay rate of average entropy of
            discriminator, only used for recording statistics
        entropy_coef (float): Weight coefficient of discriminator for entropy
            bonus [0, inf)
        discriminator_value_offset (float): Constant added to discriminator's
            output to avoid generating huge rewards.
        noisy_label (bool): Add a noise on true/fake on training
        noisy_label_range (float): The range of perturbation on the noisy label
            setting
        gpu (int): GPU device id if not None nor negative
    """
    saved_attributes = ('model', 'optimizer', 'obs_normalizer')

    def __init__(self, model, optimizer, obs_normalizer=None,
                 update_interval=3072, minibatch_size=3072, epochs=1,
                 entropy_coef=1e-3, loss_decay=0.99, entropy_decay=0.99,
                 discriminator_value_offset=1e-8, noisy_label=False,
                 noisy_label_range=0.3, gpu=None):
        self.model = model
        self.optimizer = optimizer
        self.obs_normalizer = obs_normalizer

        if gpu is not None and gpu >= 0:
            cuda.get_device_from_id(gpu).use()
            self.model.to_gpu(device=gpu)
            if obs_normalizer is not None:
                self.obs_normalizer.to_gpu(device=gpu)
        self.xp = self.model.xp

        self.epochs = epochs
        self.update_interval = update_interval
        self.minibatch_size = minibatch_size
        self.epochs = epochs
        self.entropy_coef = entropy_coef
        self.loss_decay = loss_decay
        self.entropy_decay = entropy_decay
        self.average_loss = 0.
        self.accuracy_gen = 0.
        self.accuracy_exp = 0.
        self.average_entropy = 0.
        self.discriminator_value_offset = discriminator_value_offset
        self.noisy_label = noisy_label
        self.noisy_label_range = noisy_label_range
        self._reset_trajectories()

    def get_reward(self, obs, action):
        obs = self.xp.array(obs, dtype=self.xp.float32)
        action = self.xp.array(action, dtype=self.xp.float32)
        return self.get_batch_reward(self.xp.expand_dims(obs, axis=0),
                                     self.xp.expand_dims(action, axis=0))[0]

    def get_reward_and_train(self, obs, action, expert_obs, expert_action):
        obs = self.xp.array(obs, dtype=self.xp.float32)
        action = self.xp.array(action, dtype=self.xp.float32)
        expert_obs = self.xp.array(expert_obs, dtype=self.xp.float32)
        expert_action = self.xp.array(expert_action, dtype=self.xp.float32)
        reward = self.get_reward(obs, action)
        self.trajectories_fake_obs.append(obs)
        self.trajectories_fake_action.append(action)
        self.trajectories_true_obs.append(expert_obs)
        self.trajectories_true_action.append(expert_action)
        self._update_if_dataset_is_ready()
        return reward

    def get_batch_reward(self, batch_obs, batch_action):
        batch_obs = self.xp.array(batch_obs, dtype=self.xp.float32)
        batch_action = self.xp.array(batch_action, dtype=self.xp.float32)
        if self.obs_normalizer is not None:
            batch_obs = self.obs_normalizer(batch_obs, update=False)
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            infer = self.model(batch_obs, batch_action)
            return chainer.cuda.to_cpu(
                -F.log(1
                       - F.sigmoid(infer)
                       + self.discriminator_value_offset).array)[:, 0]

    def get_batch_reward_and_train(self, batch_obs, batch_action,
                                   batch_expert_obs, batch_expert_action):
        batch_obs = self.xp.array(batch_obs, dtype=self.xp.float32)
        batch_action = self.xp.array(batch_action, dtype=self.xp.float32)
        batch_reward = self.get_batch_reward(batch_obs, batch_action)
        self.trajectories_fake_obs += list(batch_obs)
        self.trajectories_fake_action += list(batch_action)
        self.trajectories_true_obs += list(batch_expert_obs)
        self.trajectories_true_action += list(batch_expert_action)
        self._update_if_dataset_is_ready()
        return batch_reward

    def _update_if_dataset_is_ready(self):
        if len(self.trajectories_fake_obs) >= self.update_interval:
            self._update()
            self._reset_trajectories()

    def _loss(self, fake_batch_obs, fake_batch_action,
              true_batch_obs, true_batch_action):
        if self.obs_normalizer is not None:
            normalized_obs = self.obs_normalizer(fake_batch_obs, update=False)
            infer_fake = self.model(normalized_obs, fake_batch_action)
        else:
            infer_fake = self.model(fake_batch_obs, fake_batch_action)
        if self.noisy_label:
            n = fake_batch_obs.shape[0]
            fake_loss = -F.average(
                F.log(F.absolute(1 - (self.xp.random.rand(n)
                                      * self.noisy_label_range)
                                 - F.sigmoid(infer_fake))
                      + self.discriminator_value_offset))
        else:
            fake_loss = -F.average(F.log(1
                                         - F.sigmoid(infer_fake)
                                         + self.discriminator_value_offset))

        if self.obs_normalizer is not None:
            normalized_obs = self.obs_normalizer(true_batch_obs, update=True)
            infer_true = self.model(normalized_obs, true_batch_action)
        else:
            infer_true = self.model(true_batch_obs, true_batch_action)
        if self.noisy_label:
            n = true_batch_obs.shape[0]
            true_loss = -F.average(
                F.log(F.absolute(1 - (self.xp.random.rand(n)
                                      * self.noisy_label_range)
                                 - F.sigmoid(infer_true))
                      + self.discriminator_value_offset))
        else:
            true_loss = -F.average(F.log(F.sigmoid(infer_true)
                                         + self.discriminator_value_offset))

        entropy = (self._get_entropy(infer_fake) / 2
                   + self._get_entropy(infer_true) / 2)
        loss = (fake_loss + true_loss
                - entropy * self.entropy_coef)

        # Update stats
        self.accuracy_gen = np.average(
            chainer.cuda.to_cpu(infer_fake.array) < 0)
        self.accuracy_exp = np.average(
            chainer.cuda.to_cpu(infer_true.array) > 0)
        self.average_entropy *= self.entropy_decay
        self.average_entropy += (1.0 - self.entropy_decay) * chainer.cuda.to_cpu(entropy.array)  # noqa
        self.average_loss *= self.loss_decay
        self.average_loss += (1.0 - self.loss_decay) * \
            chainer.cuda.to_cpu(loss.array)

        return loss

    def _update(self):
        trajectory_size = len(self.trajectories_fake_obs)
        fake_iter = chainer.iterators.SerialIterator(
            np.random.permutation(np.arange(trajectory_size)),
            self.minibatch_size)
        true_iter = chainer.iterators.SerialIterator(
            np.random.permutation(np.arange(trajectory_size)),
            self.minibatch_size)
        # convert to numpy array
        trajectories_fake_obs = self.xp.array(self.trajectories_fake_obs,
                                              dtype=self.xp.float32)
        trajectories_fake_action = self.xp.array(self.trajectories_fake_action,
                                                 dtype=self.xp.float32)
        trajectories_true_obs = self.xp.array(self.trajectories_true_obs,
                                              dtype=self.xp.float32)
        trajectories_true_action = self.xp.array(self.trajectories_true_action,
                                                 dtype=self.xp.float32)
        while fake_iter.epoch < self.epochs:
            fake_batch_keys = np.array(fake_iter.__next__())
            fake_batch_obs = trajectories_fake_obs[fake_batch_keys]
            fake_batch_action = trajectories_fake_action[fake_batch_keys]
            true_batch_keys = np.array(true_iter.__next__())
            true_batch_obs = trajectories_true_obs[true_batch_keys]
            true_batch_action = trajectories_true_action[true_batch_keys]
            self.optimizer.update(
                lambda: self._loss(fake_batch_obs, fake_batch_action,
                                   true_batch_obs, true_batch_action))

    def _reset_trajectories(self):
        self.trajectories_fake_obs = []
        self.trajectories_fake_action = []
        self.trajectories_true_obs = []
        self.trajectories_true_action = []

    def _get_entropy(self, values):
        return F.average((-values * F.log2(F.sigmoid(values)
                                           + self.discriminator_value_offset)
                         - (1 - values) * F.log2(1
                                                 - F.sigmoid(values)
                                                 + self.discriminator_value_offset)))  # NOQA

    def get_statistics(self):
        return [
            ('accuracy_gen', self.accuracy_gen),
            ('accuracy_exp', self.accuracy_exp),
            ('average_loss', self.average_loss),
            ('average_entropy', self.average_entropy)
        ]


class GAIL(AttributeSavingMixin, BatchAgent):
    """Generative Adversarial Imination Learning
    See https://arxiv.org/abs/1606.03476
    Args:
        policy (chainerrl.Agent): Policy
        discriminator (Discriminator): Discriminator
        experts (ExpertDataset): Expert trajectory
        original_reward_weight(float): If specified, policy combines
            discriminator's output and original reward with the weight.
    """
    saved_attributes = ('policy', 'discriminator')

    def __init__(self, policy, discriminator, experts,
                 original_reward_weight=0.0):
        self.policy = policy
        self.discriminator = discriminator
        self.experts = experts
        self.last_episode = []
        self.last_observation = None
        self.last_action = None
        self.last_discriminator_value = 0
        self.t = 0
        self.update_count = 0
        self.original_reward_weight = original_reward_weight

        self.batch_last_state = None
        self.batch_last_action = None

        self.xp = self.discriminator.model.xp

    def act(self, obs):
        obs = self.xp.array(obs, dtype=self.xp.float32)
        return self.policy.act(obs)

    def act_and_train(self, obs, reward):
        obs = self.xp.array(obs, dtype=self.xp.float32)
        action = self.policy.act_and_train(
            obs,
            (self.last_discriminator_value
             + reward * self.original_reward_weight))
        self.t += 1
        self.last_observation = obs
        self.last_action = action
        expert_obs, expert_action, _, _, _ = self.experts.sample()
        self.last_discriminator_value = \
            self.discriminator.get_reward_and_train(
                obs, action, expert_obs, expert_action)
        return action

    def stop_episode_and_train(self, obs, reward, done):
        assert self.last_observation is not None
        assert self.last_action is not None

        obs = self.xp.array(obs, dtype=self.xp.float32)
        self.t += 1
        self.policy.stop_episode_and_train(obs,
                                           self.last_discriminator_value,
                                           done)
        self.stop_episode()

    def stop_episode(self):
        self.last_observation = None
        self.last_action = None
        self.last_discriminator_value = 0
        self.policy.stop_episode()

    def batch_act(self, batch_obs):
        batch_obs = self.xp.array(batch_obs, dtype=self.xp.float32)
        batch_action = self.policy.batch_act(batch_obs)
        return batch_action

    def batch_act_and_train(self, batch_obs):
        batch_obs = self.xp.array(batch_obs, dtype=self.xp.float32)
        batch_action = self.policy.batch_act_and_train(batch_obs)
        self.batch_last_state = batch_obs
        self.batch_last_action = batch_action
        return batch_action

    def batch_observe(self, batch_obs, batch_reward, batch_done, batch_reset):
        pass

    def batch_observe_and_train(self, batch_obs, batch_reward,
                                batch_done, batch_reset):
        batch_obs = self.xp.array(batch_obs, dtype=self.xp.float32)
        batch_discriminator_values = []
        for obs, action, orig_reward in zip(
                self.batch_last_state, self.batch_last_action, batch_reward):
            expert_obs, expert_action, _, _, _ = self.experts.sample()
            reward = self.discriminator.get_reward_and_train(
                obs, action, expert_obs, expert_action) \
                + orig_reward * self.original_reward_weight
            batch_discriminator_values.append(reward)

        self.policy.batch_observe_and_train(batch_obs,
                                            tuple(batch_discriminator_values),
                                            batch_done, batch_reset)

    def get_statistics(self):
        return [
            ('policy_' + name, loss)
            for name, loss in self.policy.get_statistics()
        ] + [
            ('discriminator_' + name, loss)
            for name, loss in self.discriminator.get_statistics()
        ]
