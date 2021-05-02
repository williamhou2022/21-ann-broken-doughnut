import numpy as np
import math
import random

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def dsigmoid(y):
	return y * (1 - y)

def mapList(list, func):
	vfunc = np.vectorize(func)
	return vfunc(list)

def toMatrix(list):
	list = np.array([list])
	return np.transpose(list)

class MultiLayerPerceptron1:
	
	def __init__(self, rules, *layers_list):
		self.learning_rate = rules[0]
		self.max_error = rules[1]
		self.max_iterations = rules[2]

		self.layers = len(layers_list) - 1
		self.all_weights = []
		self.all_biases = []
		for i in range(self.layers):
			self.all_weights.append(np.random.rand(layers_list[i+1], layers_list[i]) * 2 - 1)
			self.all_biases.append(np.random.rand(layers_list[i+1], 1) * 2 - 1)

	def activatePerceptron(self, inputs_array):
		values = toMatrix(inputs_array)

		for i in range(self.layers):
			values = mapList(np.matmul(self.all_weights[i], values) + self.all_biases[i], sigmoid)

		return values

	def train_one_iteration(self, inputs_list, targets_list):
		inputs = toMatrix(inputs_list)
		targets = toMatrix(targets_list)

		values_list = []
		values_list.append(inputs)
		for i in range(self.layers):
			values_list.append(mapList(np.matmul(self.all_weights[i], values_list[i]) + self.all_biases[i], sigmoid))

		errors = 0
		first_iteration = True
		for i in reversed(range(self.layers)):
			if first_iteration:
				errors = targets - values_list[i+1]
				output_errors = errors
				first_iteration = False
			else:
				errors = np.matmul(np.transpose(self.all_weights[i+1]), errors)
			gradient = self.learning_rate * mapList(values_list[i+1], dsigmoid) * errors
			self.all_weights[i] += np.matmul(gradient, np.transpose(values_list[i]))
			self.all_biases[i] += gradient

		return np.sum(np.absolute(output_errors))

	def train(self, inputs_array, targets_array):
		inputs = np.array(inputs_array)
		targets = np.array(targets_array)

		# Stochastic gradient descent
		for iterations in range(self.max_iterations):
			i = random.randint(0, len(inputs) - 1)
			error = self.train_one_iteration(inputs[i], targets[i])

			print("iterations: " + str(iterations))
			print("error: " + str(error))

		# Batch gradient descent with batch of 1
		# for iterations in range(self.max_iterations):
		#     error_sum = 0
		#     for i in range(len(inputs)):
		#         error_sum += self.train_one_iteration(inputs[i], targets[i])

		#     error_sum /= len(inputs)
		#     print("epoch: " + str(iterations))
		#     print("error sum: "+ str(error_sum))

		#     if(error_sum <= self.max_error):
		#         break