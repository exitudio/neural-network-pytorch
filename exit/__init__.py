from .optimizers import GradientDescent
from .initializers import Constant
import torch


class NeuralNetwork:
    def __init__(self, optimizer=GradientDescent(), initializer=Constant(0.1)):
        self._network = []
        self._optimizer = optimizer
        self._initializer = initializer
        self._is_init = False

    def sequence(self, *args):
        self._network = args

    def train(self, input, expected_output, loss_function, learning_rate=0.1):
        # init weight
        if not self._is_init:
            num_input = input.shape[1]
            for _, layer in enumerate(self._network):
                layer.init_weights(
                    num_input, self._optimizer, self._initializer)
                num_input = layer.num_output if hasattr(
                    layer, 'num_output') else num_input
            self._is_init = True

        # feed forward
        output_from_layer = input
        for _, layer in enumerate(self._network):
            output_from_layer = layer.feed_forward(output_from_layer, is_predict=False)

        # loss
        loss = loss_function(expected_output, output_from_layer)
        loss.backward(retain_graph=True)
        # back prop
        for _, layer in reversed(list(enumerate(self._network))):
            layer.back_prop(learning_rate)
        return {
            'output_from_layer': output_from_layer,
            'loss': loss
        }

    def predict(self, input):
        output_from_layer = input
        for _, layer in enumerate(self._network):
            output_from_layer = layer.feed_forward(output_from_layer, is_predict=True)
        return output_from_layer
