import sys
import chainer
import chainer.functions as F
from pip._internal.operations import freeze
from logging import getLogger

logger = getLogger(__name__)


def log_versions():
    logger.info(sys.version)  # Python version
    logger.info(','.join(freeze.freeze()))  # pip freeze


def ordinal_logit_function(x, axis=1, eps=1e-8):
    '''Ordinal discrete logits
    See https://arxiv.org/pdf/1901.10500.pdf

    Args:
        x (:class:`~chainer.Variable` or :ref`ndarray`):
            Elements to calculate the ordinal logits
        eps (float):
            Epsilon value to avoid log(0)
    Returns:
        ~chainer.Variable: Output variable.
    '''
    s = F.sigmoid(x)
    cumsum_left = F.cumsum(F.log(s + eps), axis=axis)
    cumsum_right = F.cumsum(F.log(1 - s + eps), axis=axis)
    sum_right = F.sum(F.log(1 - s + eps), axis=axis, keepdims=True)
    y = cumsum_left + sum_right - cumsum_right
    return y


# test ordinal_logit_function
if __name__ == '__main__':
    import numpy as np
    inputs = chainer.Variable(np.array([[0, 0, 0], [0, 1e3, 0]], dtype=np.float32))
    print(ordinal_logit_function(inputs))
