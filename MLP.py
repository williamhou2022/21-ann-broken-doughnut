import numpy as np
import math
import random

# https://youtube.com/playlist?list=PLRqwX-V7Uu6aCibgK1PTWWu9by6XFdCfh
# https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi

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

class MultiLayerPerceptron:
    
    def __init__(self, inputs, hidden, outputs, rules):
        self.ip = inputs
        self.hd = hidden
        self.op = outputs

        self.ih_weights = np.random.rand(self.hd, self.ip) * 2 - 1
        self.ho_weights = np.random.rand(self.op, self.hd) * 2 - 1

        self.h_bias = np.random.rand(self.hd, 1) * 2 - 1
        self.o_bias = np.random.rand(self.op, 1) * 2 - 1

        self.learning_rate = rules[0]
        self.max_error = rules[1]
        self.max_iterations = rules[2]

    def activatePerceptron(self, inputs_array):
        inputs = toMatrix(inputs_array)

        hidden = np.matmul(self.ih_weights, inputs)
        hidden += self.h_bias
        hidden = mapList(hidden, sigmoid)

        outputs = np.matmul(self.ho_weights, hidden)
        outputs += self.o_bias
        outputs = mapList(outputs, sigmoid)

        return outputs

    def train_one_iteration(self, inputs_list, targets_list):
        inputs = toMatrix(inputs_list)
        targets = toMatrix(targets_list)

        # calculate hidden layer values
        hidden = np.matmul(self.ih_weights, inputs)
        hidden += self.h_bias
        hidden = mapList(hidden, sigmoid)
        # calculate output layer values
        outputs = np.matmul(self.ho_weights, hidden)
        outputs += self.o_bias
        outputs = mapList(outputs, sigmoid) 

        output_errors = targets - outputs
        output_gradient = self.learning_rate * mapList(outputs, dsigmoid) * output_errors
        ho_weights_deltas = np.matmul(output_gradient, np.transpose(hidden))
        self.ho_weights += ho_weights_deltas
        self.o_bias += output_gradient

        hidden_errors = np.matmul(np.transpose(self.ho_weights), output_errors)
        hidden_gradient = self.learning_rate * mapList(hidden, dsigmoid) * hidden_errors
        ih_weights_deltas = np.matmul(hidden_gradient, np.transpose(inputs))
        self.ih_weights += ih_weights_deltas
        self.h_bias += hidden_gradient

        return np.sum(np.absolute(output_errors))

    def train(self, inputs_array, targets_array):
        inputs = np.array(inputs_array)
        targets = np.array(targets_array)

        # Stochastic gradient descent
        for iterations in range(self.max_iterations):
            i = random.randint(0, len(inputs) - 1)
            error = self.train_one_iteration(inputs[i], targets[i])

            print(iterations)
            print(error)

        # Batch gradient descent with batch of 1
        # for iterations in range(self.max_iterations):
        #     error_sum = 0
        #     for i in range(len(inputs)):
        #         error_sum += self.train_one_iteration(inputs[i], targets[i])

        #     error_sum /= len(inputs)
        #     print("epoch: " + str(iterations))
        #     print("error sum: + str(error_sum))

        #     if(error_sum <= self.max_error):
        #         break