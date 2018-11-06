from abc import ABC, abstractmethod
from .initializers import Constant
from .optimizers import Momentum
from .constants import EPSILON
import torch


class Layer(ABC):
    @abstractmethod
    def init_weights(self, num_input, optimizer, initializer):
        pass

    @abstractmethod
    def feed_forward(self, input, is_predict):
        pass

    @abstractmethod
    def back_prop(self, learning_rate):
        pass


class Dense(Layer):
    def __init__(self, num_output):
        self._num_output = num_output

    def init_weights(self, num_input, optimizer, initializer):
        # init weights
        self._weights = initializer(num_input, self._num_output)
        self._bias = initializer(1, self._num_output)
        # init optimizer
        self._optimizer_w = optimizer.generate_optimizer(self._weights.shape)
        self._optimizer_b = optimizer.generate_optimizer(self._bias.shape)

    def feed_forward(self, input, is_predict):
        return input.mm(self._weights) + self._bias

    def back_prop(self, learning_rate):
        with torch.no_grad():
            self._weights -= learning_rate * \
                self._optimizer_w.get_velocity(self._weights.grad)
            self._bias -= learning_rate * \
                self._optimizer_b.get_velocity(self._bias.grad)
            self._weights.grad.zero_()
            self._bias.grad.zero_()

    @property
    def num_output(self):
        return self._num_output


class BatchNorm(Layer):
    """
    I don't know why batch norm does worse than normal layer
    Batch norm backprop
    https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
    """

    def init_weights(self, num_output, optimizer, initializer):
        self._batch_norm_G = initializer(1, num_output)
        self._batch_norm_B = initializer(1, num_output)
        # init optimizer
        self._optimizer_G = optimizer.generate_optimizer((1, num_output))
        self._optimizer_B = optimizer.generate_optimizer((1, num_output))
        self._optimizer_mean = Momentum().generate_optimizer(
            (1, num_output))
        self._optimizer_variance = Momentum().generate_optimizer(
            (1, num_output))

    def feed_forward(self, z, is_predict):
        current_mean = z.mean(dim=0)
        with torch.no_grad():
            mean = self._optimizer_mean._velocity if is_predict else self._optimizer_mean.get_velocity(
                current_mean)

        diff_mean = z-mean

        current_variance = torch.pow(diff_mean, 2).mean(dim=0)
        with torch.no_grad():
            variance = self._optimizer_variance._velocity if is_predict else self._optimizer_variance.get_velocity(
                current_variance)

        z_norm = diff_mean / torch.sqrt(variance + EPSILON)
        output = self._batch_norm_G * z_norm + self._batch_norm_B
        return output

    def back_prop(self, learning_rate):
        with torch.no_grad():
            self._batch_norm_G -= learning_rate * \
                self._optimizer_G.get_velocity(self._batch_norm_G.grad)
            self._batch_norm_B -= learning_rate * \
                self._optimizer_G.get_velocity(self._batch_norm_B.grad)
            self._batch_norm_G.grad.zero_()
            self._batch_norm_B.grad.zero_()
