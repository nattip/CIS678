# Written by Natalie Tipton
# March 31, 20202
# CIS 678 - Machine Learning
# Dr. Greg Wolffe
# Neural Network

import numpy as np
from numpy import exp, array, random, dot
from collections import Counter
import matplotlib.pyplot as plt


# class for all neural net training and testing
class NeuralNet:
    def __init__(self):
        # see random generator for weight creation
        random.seed(1)

        # declare the number of hidden nodes and the learning rate
        self.n_hidden = 50
        self.lr = 0.0001

        # generate random weights and bias for all hidden nodes
        self.weights_hidden = random.normal(0, 1, (inputs.shape[1], self.n_hidden))
        self.bias_hidden = np.zeros(self.n_hidden)

        # decleare number of output nodes and generate weights and bias for them
        self.n_output = n_outputs
        self.weights_output = random.normal(0, 1, (self.n_hidden, self.n_output))
        self.bias_output = np.zeros(self.n_output)

    # encodes the target values from the dataset
    def encode_targets(self, targets, n_classes):
        onehot = np.zeros((n_classes, targets.shape[0]))
        for i, v in enumerate(targets.astype(int)):
            onehot[v, i] = 1.0
        return onehot.T

    # calculates the simgoid activation function for a node and
    # clips the value so it can't explode
    def sigmoid(self, x):
        return 1 / (1 + exp(-np.clip(x, -250, 250)))

    # returns the derivative of the sigmoid activation function
    def deriv_sigmoid(self, x):
        return x * (1 - x)

    # finds the dot product of all inputs and weights to a node and
    # calculates the sigmoid activation value of that node
    # for a hidden layer and the output layer
    def forward(self, inputs):
        hidden_out = dot(inputs, self.weights_hidden) + self.bias_hidden
        hidden_sigmoid = self.sigmoid(hidden_out)

        output_out = dot(hidden_sigmoid, self.weights_output) + self.bias_output
        output_sigmoid = self.sigmoid(output_out)

        return hidden_sigmoid, output_sigmoid

    # trains the nerual net by adjusting weights and bias
    # values at the hidden and output layers based on the
    # amount of error in current values
    def train(self, training_inputs, targets, iterations):
        targets = self.encode_targets(targets, 10)
        for iteration in range(iterations):
            # find sigmoid activation for both layers
            hidden_sigmoid, output_sigmoid = self.forward(training_inputs)

            # calculate the error in output value
            delta_output = output_sigmoid - targets

            # backpropagate to find error in hidden layer
            deriv_sig_hidden = self.deriv_sigmoid(hidden_sigmoid)
            delta_hidden = dot(delta_output, self.weights_output.T) * deriv_sig_hidden

            # find gradient of hidden layer for weights and bias
            grad_weight_hidden = dot(training_inputs.T, delta_hidden)
            grad_bias_hidden = np.sum(delta_hidden, axis=0)

            # find gradient of output layer for weights and bias
            grad_weight_output = dot(hidden_sigmoid.T, delta_output)
            grad_bias_output = np.sum(delta_output, axis=0)

            # adjust all weights and bias by a function of the learning rate
            # and the gradient calculated
            self.weights_hidden -= self.lr * grad_weight_hidden
            self.bias_hidden -= self.lr * grad_bias_hidden

            self.weights_output -= self.lr * grad_weight_output
            self.bias_output -= self.lr * grad_bias_output

    # runs test data through the neural net and uses argmax to return a prediction
    def test(self, testing_inputs):
        hidden_sigmoid_test, output_sigmoid_test = self.forward(testing_inputs)

        prediction = np.argmax(output_sigmoid_test, axis=1)
        return prediction


if __name__ == "__main__":

    # read in training data and separate the inputs
    # from the target values
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

    # turn array of strings into array of float64
    inputs = array(inputs, dtype=np.float64)
    targets = array(targets, dtype=np.float64)

    # determine how many possible classes there are
    class_labels = np.unique(targets)

    # find the number of classes and number of inputs
    n_outputs = len(class_labels)
    n_features = inputs.shape[1]

    # create and train the neural net
    nn = NeuralNet()
    nn.train(inputs, targets, 10000)

    # read in test data and separate targets from inputs
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

    # turn array of strings into array of float64
    inputs_test = array(inputs_test, dtype=np.float64)
    targets_test = array(targets_test, dtype=np.float64)

    # obtain array of predictions from the neural net
    predictions = nn.test(inputs_test)

    # initalize number correct and number incorrect
    correct = 0
    incorrect = 0

    # compare all predictions to the targets from the test set
    # and count how many were right/wrong
    for value in range(len(predictions)):
        if predictions[value] == targets_test[value]:
            correct += 1
        else:
            incorrect += 1

    # calculate the accuracy of the neural net
    accuracy = 100 * (correct / (correct + incorrect))

    percent_error = {}
    for value in class_labels:
        # exp-theo/theo
        percent_error[value] = round(
            100
            * (Counter(predictions)[value] - Counter(targets_test)[value])
            / Counter(targets_test)[value],
            2,
        )

    # print results
    print(f"Number correct: {correct}")
    print(f"Number incorrect: {incorrect}")
    print(f"Accuracy: {round(accuracy, 2)}")
    print(f"Number of predictions for each digit: {Counter(predictions)}")
    print(f"Actual number of digits in test set: {Counter(targets_test)}")

    # create bar plot of percent errors for each digit
    bar = plt.bar(*zip(*percent_error.items()))

    # change color of 2 digits with highest errors
    bar[7].set_color("r")
    bar[1].set_color("y")

    # label plot
    plt.ylabel("Percent Error")
    plt.xlabel("Digits")
    plt.title("Percent Error for Predictions")
    x_ticks = np.arange(0, 10, 1)
    plt.xticks(x_ticks)

    # show plot
    plt.show()
