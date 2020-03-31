# Written by Natalie Tipton
# March 31, 20202
# CIS 678 - Machine Learning
# Dr. Greg Wolffe
# Neural Network

import numpy as np
from numpy import exp, array, random, dot
from collections import Counter


class NeuralNet:
    def __init__(self):
        random.seed(1)

        self.n_hidden = 10
        self.lr = 0.0001

        self.weights_hidden = random.normal(
            0, 1, (inputs.shape[1], self.n_hidden)
        )
        self.bias_hidden = np.zeros(self.n_hidden)

        self.n_output = n_outputs
        print(self.n_output, n_outputs)
        self.weights_output = random.normal(
            0, 1, (self.n_hidden, self.n_output)
        )
        self.bias_output = np.zeros(self.n_output)

    def encode_targets(self, targets, n_classes):
        onehot = np.zeros((n_classes, targets.shape[0]))
        for i, v in enumerate(targets.astype(int)):
            onehot[v, i] = 1.0
        return onehot.T

    def sigmoid(self, x):
        return 1 / (1 + exp(-np.clip(x, -250, 250)))

    def deriv_sigmoid(self, x):
        return x * (1 - x)

    def forward(self, inputs):
        hidden_out = dot(inputs, self.weights_hidden) + self.bias_hidden
        hidden_sigmoid = self.sigmoid(hidden_out)

        output_out = (
            dot(hidden_sigmoid, self.weights_output) + self.bias_output
        )
        output_sigmoid = self.sigmoid(output_out)

        return hidden_sigmoid, output_sigmoid

    def train(self, training_inputs, targets, iterations):
        targets = self.encode_targets(targets, 10)
        for iteration in range(iterations):
            hidden_sigmoid, output_sigmoid = self.forward(training_inputs)

            delta_output = output_sigmoid - targets

            deriv_sig_hidden = self.deriv_sigmoid(hidden_sigmoid)
            delta_hidden = (
                dot(delta_output, self.weights_output.T) * deriv_sig_hidden
            )

            grad_weight_hidden = dot(training_inputs.T, delta_hidden)
            grad_bias_hidden = np.sum(delta_hidden, axis=0)
            # print(grad_weight_hidden, "BIAS", grad_bias_hidden)

            grad_weight_output = dot(hidden_sigmoid.T, delta_output)
            grad_bias_output = np.sum(delta_output, axis=0)
            # print(grad_weight_output, "BIAS", grad_bias_output)

            self.weights_hidden -= self.lr * grad_weight_hidden
            self.bias_hidden -= self.lr * grad_bias_hidden

            self.weights_output -= self.lr * grad_weight_output
            self.bias_output -= self.lr * grad_bias_output

    def test(self, testing_inputs):
        hidden_sigmoid_test, output_sigmoid_test = self.forward(testing_inputs)

        prediction = np.argmax(output_sigmoid_test, axis=1)
        return prediction


if __name__ == "__main__":

    with open("./digits-training.data") as f:
        lines = f.readlines()
        flag = 0

        for line in lines:
            if not flag:
                targets = array([line.rstrip().split(" ").pop()])
                inputs = array([line.rstrip().split(" ")[:-1]])
                flag = 1
            else:
                targets = np.append(
                    targets, array([line.rstrip().split(" ").pop()])
                )
                inputs = np.concatenate(
                    (inputs, array([line.rstrip().split(" ")[:-1]]))
                )

    inputs = array(inputs, dtype=np.float64)
    targets = array(targets, dtype=np.float64)

    class_labels = np.unique(targets)
    n_outputs = len(class_labels)
    n_features = inputs.shape[1]

    nn = NeuralNet()
    nn.train(inputs, targets, 100)

    with open("./digits-test.data") as f:
        lines = f.readlines()
        flag = 0

        for line in lines:
            if not flag:
                targets_test = array([line.rstrip().split(" ").pop()])
                inputs_test = array([line.rstrip().split(" ")[:-1]])
                flag = 1
            else:
                targets_test = np.append(
                    targets_test, array([line.rstrip().split(" ").pop()])
                )
                inputs_test = np.concatenate(
                    (inputs_test, array([line.rstrip().split(" ")[:-1]]))
                )

    inputs_test = array(inputs_test, dtype=np.float64)
    targets_test = array(targets_test, dtype=np.float64)

    print(Counter(nn.test(inputs_test)))
