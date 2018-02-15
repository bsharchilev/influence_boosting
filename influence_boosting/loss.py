import numpy as np
from copy import deepcopy
from abc import ABCMeta, abstractmethod

from .util import expand_shapes_to_array


def _array_expanding_wrapper(treat_input_as_row=True):
    def __array_expanding_wrapper(func):
        def cast_inputs_to_arrays_and_compute(self, target, y, **kwargs):
            target_array, was_target_expanded = expand_shapes_to_array(target, first_dim=treat_input_as_row)
            y_array, was_y_expanded = expand_shapes_to_array(y, first_dim=treat_input_as_row)
            assert was_target_expanded == was_y_expanded, '%s %s' % (was_target_expanded, was_y_expanded)
            assert target_array.shape == y_array.shape, '%s %s' % (target_array.shape, y_array.shape)

            result = func(self, target_array, y_array, **kwargs)
            if not was_y_expanded:
                return result
            else:
                if treat_input_as_row:
                    return result[0]
                else:
                    return result[:, 0]
        return cast_inputs_to_arrays_and_compute
    return __array_expanding_wrapper


class Loss(object):
    __metaclass__ = ABCMeta

    @_array_expanding_wrapper()
    @abstractmethod
    def __call__(self, target, y):
        """
        For np.ndarrays target (nxd or d) and y (nxd or d), calculate the vector of losses (n or 1)
        :param target: targets.
        :param y: predicted values.
        :return: losses
        """
        pass

    @_array_expanding_wrapper()
    @abstractmethod
    def gradient(self, target, y):
        return np.array([])

    @_array_expanding_wrapper()
    @abstractmethod
    def hessian(self, target, y):
        # 3D matrix: (nxdxd)
        return np.array([])

    @_array_expanding_wrapper()
    @abstractmethod
    def third(self, target, y):
        return np.array([])

    @_array_expanding_wrapper()
    def ihvp(self, target, y, l2_reg=0):
        return np.linalg.solve(self.hessian(target, y) + l2_reg * _batched_diag(np.ones_like(target)),
                               self.gradient(target, y))

    @property
    @abstractmethod
    def is_multidimensional(self):
        return False


class CrossEntropyLoss(Loss):
    # TODO: also support scalar losses later
    @_array_expanding_wrapper()
    def __call__(self, target, y):
        logsoftmax = y - _logsumexp(y, keepdims=True)
        return -np.einsum('ij,ij->i', target, logsoftmax)

    @_array_expanding_wrapper()
    def gradient(self, target, y):
        return _softmax(y) - target

    @_array_expanding_wrapper()
    def hessian(self, target, y):
        y_softmax = _softmax(y)
        common_part = -np.einsum('ni,nj->nij', y_softmax, y_softmax)
        diagonal_part = _batched_diag(y_softmax)
        return common_part + diagonal_part

    @property
    def is_multidimensional(self):
        return True


class BinaryCrossEntropyLoss(Loss):
    def __call__(self, target, y):
        _y = deepcopy(y)
        target_is_zero = ~np.isclose(target, 1)
        _y[target_is_zero] = -_y[target_is_zero]
        assert len(_y.shape) == 1, _y.shape
        return _softplus(-_y)

    def gradient(self, target, y):
        return _sigmoid(y) - target

    def hessian(self, target, y):
        return _sigmoid(y) * (1 - _sigmoid(y))

    def third(self, target, y):
        return _sigmoid(y) * (1 - _sigmoid(y)) * (1 - 2 * _sigmoid(y))

    @property
    def is_multidimensional(self):
        return False


class MSELoss(Loss):
    def __call__(self, target, y):
        return (y - target) ** 2

    def gradient(self, target, y):
        return 2 * (y - target)

    def hessian(self, target, y):
        return 2 * np.ones_like(target)

    def third(self, target, y):
        return np.zeros_like(target)

    @property
    def is_multidimensional(self):
        return False


def _sigmoid(x):
    return 1. / (1. + np.exp(-x))


def _softmax(y):
    centered_exponent = np.exp(y - np.max(y, axis=1, keepdims=True))
    return centered_exponent / np.sum(centered_exponent, axis=1, keepdims=True)


def _softplus(x, limit=30):
    result = deepcopy(x)
    no_overflow = x <= limit
    result[no_overflow] = np.log(1.0 + np.exp(result[no_overflow]))
    return result


def _logsumexp(y, keepdims):
    maximum = np.max(y, axis=1, keepdims=True)
    result_keepdims = maximum + np.log(np.sum(np.exp(y - maximum), axis=1, keepdims=True))
    if keepdims:
        return result_keepdims
    else:
        return result_keepdims[:, 0]


def _batched_diag(matrix):
    return np.stack([np.diag(row) for row in matrix], axis=0)
