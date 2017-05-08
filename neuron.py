import numpy as np
import random


def sigmoid(s):
    return 1. / (1. + np.exp(-s))


def sigmoid_derivative(s):
    sigmoid_value = sigmoid(s)
    return sigmoid_value * (1. - sigmoid_value)


class Neuron(object):
    """
    This is a class for a single neuron in our Neural Net.
    In neural_network.py we have a two dimensional python list that represents our Neural Net. Each
    element in this list is of type Neuron.
    """
    def __init__(self,
                 n_inputs,
                 weights=None,
                 transfer_function=sigmoid,
                 transfer_function_derivative=sigmoid_derivative):
        """
        __init__ is the constructor of the Neuron
        param n_inputs: integer, number of inputs to the neuron from the previous layer.
        param weights: a python list. These are the weights corresponding to the inputs. Note
            that len(weights) should be equal to n_inputs+1 because of the "bias" term.
        param transfer_function: in our case this is the sigmoid function.
        transfer_function_derivative: in our case, this is the derivative of the sigmoid.

        """
        self.transfer_function = transfer_function
        if weights is None:
            self.weights = np.array([1e-5 * random.random()
                            for _ in range(n_inputs + 1)])
        else:
            assert n_inputs == len(weights) - 1
            self.weights = weights
        self.derivative_function = transfer_function_derivative

        # Values for the last feed-forward
        self.weighted_sum = None
        self.output = None
        self.inputs = None
        # Values for the last feed-backward
        self.delta = None

    def forward_propagate(self, inputs):
        self.inputs=inputs
        self.weighted_sum = self.weights.dot(inputs)       #NP
        self.output=sigmoid(self.weighted_sum)

        return self.output

    def feed_backwards(self, error):

        self.delta=sigmoid_derivative(self.weighted_sum)*error


    def update_weights(self, learning_rate):

        self.weights=self.weights         #NP
        self.weights-=learning_rate * self.delta * self.inputs
