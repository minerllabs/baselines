import numpy as np
import chainer
from distribution import MultiDimensionalSoftmaxDistribution
import chainer.functions as F


def test_multi_dimensional_softmax_distribution():
    v = chainer.Variable(np.array([[2, 2.1], [3, 3], [3, 3], [3, 3]], dtype=np.float32))
    w = chainer.Variable(np.array([[-2, -1, -0.999], [3, 3, 3], [3, 3, 3], [3, 3, 3]], dtype=np.float32))
    v2 = chainer.Variable(np.array([[3, 3]], dtype=np.float32))
    w2 = chainer.Variable(np.array([[-1, -1, 0]], dtype=np.float32))
    dist = MultiDimensionalSoftmaxDistribution([v, w])
    dist2 = MultiDimensionalSoftmaxDistribution([v2, w2])
    assert len(dist.params) == 2
    for param, d_param in zip((v, w), dist.params):
        assert np.allclose(param.array, d_param.array)
    assert dist.sample().shape == (4, 2)
    assert dist.prob(np.array([[0, 0], [0, 0], [0, 0], [1, 2]])).shape == (4,)
    assert dist.log_prob(np.array([[0, 2], [0, 0], [0, 0], [0, 0]])).shape == (4,)
    assert np.allclose(np.array([[1, 2], [0, 0], [0, 0], [0, 0]]),
                       dist.most_probable.array)
    assert dist.kl(dist2).shape == (4,)
