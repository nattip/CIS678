# Written by Natalie Tipton
# March 31, 20202
# CIS 678 - Machine Learning
# Dr. Greg Wolffe
# Neural Network

import numpy as np
from numpy import exp, array, random, dot


class neuralNet:
    def __init__(self):
        random.seed(1)

        self.weights = 2 * random.random(-1, 1) - 1

    def sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def deriv_sigmoid(self, x):
        return x*(1-x)

    def foreward(self, inputs):
        return self.sigmoid(dot(inputs, self.weights))

    def train(self, training_inputs, targets, iterations):
        for iteration in iterations:
            output = self.roreward(training_data)

            error = targets - output

            grad = dot(training_inputs.T, error * self.deriv_sigmoid(output))

            self.weights += grad

if __name__ == "__main__":
    nn = neuralNet
    inputs = 
    targets = 

    nn.train(inputs,targets,100)

    print(f"weights: {nn.weights}")


