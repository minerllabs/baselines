import chainer
from chainerrl.distribution import Distribution, SoftmaxDistribution
import chainer.functions as F
from cached_property import cached_property


def _unwrap_variable(x):
    if isinstance(x, chainer.Variable):
        return x.array
    else:
        return x


class MultiDimensionalSoftmaxDistribution(Distribution):
    def __init__(self, logits, beta=1.0, min_prob=0.0):
        self.distributions = []
        self.beta = beta
        self.min_prob = min_prob
        self.logits = logits
        for logit in logits:
            self.distributions.append(
                SoftmaxDistribution(logit, beta, min_prob))

    @cached_property
    def entropy(self):
        ent_sum = 0
        for distribution in self.distributions:
            ent_sum += distribution.entropy
        return ent_sum

    def sample(self):
        values = [distribution.sample().reshape(-1, 1)
                  for distribution in self.distributions]
        return F.concat(tuple(values), axis=-1)

    def prob(self, x):
        assert x.shape[1] == len(self.distributions)
        prob_all = 1
        for value, distribution in zip(list(F.separate(x, axis=1)),
                                       self.distributions):
            prob_all *= distribution.prob(value)
        return prob_all

    def log_prob(self, x):
        prob_all = self.prob(x)
        return F.log(prob_all)

    def copy(self):
        return MultiDimensionalSoftmaxDistribution(
            [_unwrap_variable(logit).copy() for logit in self.logits],
            self.beta, self.min_prob)

    @cached_property
    def most_probable(self):
        values = [distribution.most_probable.reshape(-1, 1)
                  for distribution in self.distributions]
        return F.concat(tuple(values), axis=-1)

    @property
    def params(self):
        return tuple(self.logits)

    def kl(self, distrib):
        kl_sum = 0
        for dist_a, dist_b in zip(self.distributions, distrib.distributions):
            kl_sum += dist_a.kl(dist_b)
        return kl_sum

    def __repr__(self):
        return 'MultiDimensionalSoftmaxDistribution({})'.format(
            [repr(distribution) for distribution in self.distributions])
