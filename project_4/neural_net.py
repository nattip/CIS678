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

        self.n_hidden = 10
        self.lr = 0.0001

        self.weights_hidden = random.normal(0, 1, (inputs.shape[1], self.n_hidden))
        self.bias_hidden = np.zeros(self.n_hidden)

        self.n_output = n_outputs
        self.weights_output = random.normal(0, 1, (self.n_hidden, self.n_output))
        self.bias_output = np.zeros(self.n_output)

    def sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def deriv_sigmoid(self, x):
        return x * (1 - x)

    def forward(self, inputs):
        hidden_out = dot(inputs, self.weights_hidden) + self.bias_hidden
        hidden_sigmoid = self.sigmoid(hidden_out)

        output_out = dot(hidden_sigmoid, self.weights_output) + self.bias_output
        output_sigmoid = self.sigmoid(output_out)

        return hidden_sigmoid, output_sigmoid

    def train(self, training_inputs, targets, iterations):
        for iteration in range(iterations):
            hidden_sigmoid, output_sigmoid = self.forward(training_inputs)

            delta_output = output_sigmoid.T - targets

            deriv_sig_hidden = self.deriv_sigmoid(hidden_sigmoid)
            delta_hidden = dot(delta_output.T, self.weights_output.T) * deriv_sig_hidden

            grad_weight_hidden = dot(training_inputs.T, delta_hidden)
            grad_bias_hidden = np.sum(delta_hidden, axis=0)

            grad_weight_output = dot(hidden_sigmoid.T, delta_output.T)
            grad_bias_output = np.sum(delta_output.T, axis=0)

            # print(grad_weight_output)
            # print(grad_weight_hidden)

            self.weights_hidden -= self.lr * grad_weight_hidden
            self.bias_hidden -= self.lr + grad_bias_hidden

            self.weights_output -= self.lr * grad_weight_output
            self.bias_output -= self.lr + grad_bias_output

            # print(self.weights_output[0][0])
            # print(self.weights_hidden[0][0])

    def test(self, testing_inputs):
        (hidden_sigmoid_test, output_sigmoid_test,) = self.forward(testing_inputs)
        # print(output_sigmoid_test.shape)
        # print(output_sigmoid_test)

        # prediction = np.argmax(output_sigmoid_test, axis=1)
        # print(prediction)
        print(output_sigmoid_test)


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
                targets = np.append(targets, array([line.rstrip().split(" ").pop()]))
                inputs = np.concatenate(
                    (inputs, array([line.rstrip().split(" ")[:-1]]))
                )

    inputs = array(inputs, dtype=np.float64)
    targets = array(targets, dtype=np.float64)

    class_labels = np.unique(targets)
    n_outputs = len(class_labels)
    n_features = inputs.shape[1]

    nn = neuralNet()
    nn.train(inputs, targets, 500)

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

    for data in range(0, len(inputs_test)):
        nn.test(inputs_test[data])
