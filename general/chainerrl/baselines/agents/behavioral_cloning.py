"""
MIT License

Copyright (c) Preferred Networks, Inc.
"""

from logging import getLogger

import minerl  # noqa: register MineRL envs as Gym envs.
import numpy as np

import chainer
from chainer import cuda
from chainer import functions as F

from chainerrl.agent import AttributeSavingMixin, Agent

logger = getLogger(__name__)


class BehavioralCloning(AttributeSavingMixin, Agent):
    """Behavioral Cloning
    Args:
        model (chainer.Chain): Model
        optimizer (chainer.Optimizer): Optimizer to train model
        experts (ExpertDataset): Expert trajectory
        minibatch_size (int): Minibatch size
        states_per_epoch (int): Number of states to use in one training
            iteration
        action_wrapper (str): Name of action wrapper
        entropy_coef (float): Weight coefficient for entropy bonus [0, inf)
        gpu (int): GPU device id if not None nor negative
    """
    saved_attributes = ('model', 'optimizer')

    def __init__(self, model, optimizer,
                 minibatch_size=128,
                 states_per_epoch=2048,
                 action_wrapper='discrete',
                 entropy_coef=0.01, gpu=None):
        if gpu is not None and gpu >= 0:
            cuda.get_device_from_id(gpu).use()
            self.model.to_gpu(device=gpu)

        self.model = model
        self.optimizer = optimizer
        self.minibatch_size = minibatch_size
        self.states_per_epoch = states_per_epoch
        self.average_loss = 1e38
        self.action_wrapper = action_wrapper
        self.entropy_coef = entropy_coef
        self.xp = self.model.xp

    def act(self, obs):
        obs = self.xp.array(obs)
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            q = chainer.cuda.to_cpu(self.model(np.expand_dims(obs, axis=0)).sample().array)
            return q[0]

    def act_and_train(self, obs, reward):
        return self.act(obs)

    def stop_episode_and_train(self, obs, reward, done):
        pass

    def stop_episode(self):
        pass

    def _loss(self, batch_obs, batch_acs):
        out = self.model(batch_obs)
        entropy = F.average(out.entropy)
        if self.action_wrapper == 'discrete':
            loss = F.softmax_cross_entropy(out.params[0], batch_acs.reshape(-1))
        elif self.action_wrapper == 'continuous':
            loss = F.mean_squared_error(out.params[0], batch_acs)
        elif self.action_wrapper == 'multi-dimensional-softmax':
            loss = 0
            for idx, logit in enumerate(out.params):
                expected = batch_acs[:, idx]
                loss += F.softmax_cross_entropy(logit, expected)
        loss -= entropy * self.entropy_coef
        return loss

    def train(self, train_obs, train_acs, _validate_obs, validate_acs,
              max_retry=100, minimum_update_delta=1e-8):
        """
        This function trains the model by training and validation dataset.
        It updates the model by training dataset until it fails improving
           the loss of validation dataset max_retry times in a row.
        """
        current_loss = 1e38
        length = len(train_obs)
        num_retry = 0
        if len(_validate_obs.shape) == 1:
            # LazyFrames
            validate_obs = self.xp.array([self.xp.array(ob) for ob in _validate_obs])
        else:
            validate_obs = _validate_obs
        while True:
            keys = np.random.permutation(length)[:self.minibatch_size]
            batch_obs = train_obs[keys]
            if len(batch_obs.shape) == 1:
                # LazyFrames
                batch_obs = self.xp.array([self.xp.array(ob) for ob in batch_obs])
            batch_acs = train_acs[keys]
            self.optimizer.update(
                lambda: self._loss(batch_obs, batch_acs))
            validate_loss = chainer.cuda.to_cpu(
                self._loss(validate_obs, validate_acs).array)
            logger.debug('Validate_loss: {}'.format(validate_loss))
            if validate_loss > current_loss - minimum_update_delta:
                num_retry += 1
            else:
                num_retry = 0
                current_loss = validate_loss
            if num_retry == max_retry:
                break
        self.average_loss = current_loss

    def get_statistics(self):
        return [('average_loss', self.average_loss)]
