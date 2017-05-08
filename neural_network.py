from neuron import Neuron
import numpy as np
from neuron import sigmoid_derivative


class NeuralNetwork(object):

	def __init__(self, n_features, neurons_per_layer):
		self.npl=np.array(neurons_per_layer)
		self.n_features=np.hstack((n_features, self.npl[:len(self.npl) - 1]))

		self.output = []
		self.neuron = []
		for i in range(len(self.npl)):
			self.neuron.append([])
			self.output.append([])
			for j in range(self.npl[i]):
				self.neuron[i].append(Neuron(self.n_features[i]))
				self.output[i].append(self.neuron[i][j].output)
		
	def forward_propagate(self, features):
		features=np.hstack((1, np.array(features)))
		for i in range(len(self.npl)):
			for j in range(self.npl[i]):
				self.output[i][j]=self.neuron[i][j].forward_propagate(features)
			features=np.hstack((1, np.array(self.output[i])))
		return self.output[-1]

	def backward_propagate(self, correct_output_vector):
		
		def error(layer_level):
			if layer_level ==len(self.npl)-1:
				out_error=np.array(self.output[-1])-np.array(correct_output_vector)
				return out_error
			else:
				
				ds_arg = np.zeros(self.npl[layer_level + 1])
				for k in range(self.npl[layer_level + 1]):

					s=self.neuron[layer_level + 1][k].weights[1:self.npl[layer_level] + 1].dot(self.output[layer_level][:self.npl[layer_level]])

					ds_arg[k] = self.neuron[layer_level + 1][k].weights[0] + s

				ds = sigmoid_derivative(ds_arg)
				
				next_error = error(layer_level + 1)
				
				out_error = []
				for i in range(self.npl[layer_level]):

					w=np.array([self.neuron[layer_level+1][k].weights[i] for k in range(self.npl[layer_level+1])])

					out_error.append(sum([w[k]*ds[k]*next_error[k] for k in range(self.npl[layer_level+1])]))
				return out_error
		
		err=[]
		for i in range(len(self.npl)):
			err.append(error(i))
		
		for i in range(len(self.npl)):
			for j in range(self.npl[i]):
				self.neuron[i][j].feed_backwards(err[i][j])

	def update_weights(self, learning_rate):
		
		for i in range(len(self.npl)):
			for j in range(self.npl[i]):
				self.neuron[i][j].update_weights(learning_rate)


	def train(self, X, Y, learning_rate=1, max_iter=2):
		
		for j in range(max_iter):

			for i in range(len(X)):
				self.forward_propagate(X[i])
				self.backward_propagate(Y[i])
				self.update_weights(learning_rate)
