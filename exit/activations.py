from .layers import Layer
import torch


class Activation(Layer):
    def init_weights(self, num_input, optimizer, initializer):
        pass

    def back_prop(self, learning_rate):
        pass


class Sigmoid(Activation):
    def feed_forward(self, input, is_predict):
        return 1 / (1 + torch.exp(-input))


class Relu(Activation):
    zero = torch.Tensor([0])
    def feed_forward(self, input, is_predict):
        self.input = input
        output = torch.max(Relu.zero, input)
        return output