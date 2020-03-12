# Written by Natalie Tipton
# March 10, 20202
# CIS 678 - Machine Learning
# Dr. Greg Wolffe
# ID3 decision tree creator

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import copy
from collections import Counter
from networkx import drawing
from typing import Tuple, List


def open_data(filename):
    # open dataset
    with open("fishing_data.txt") as f:

        # read in metaData for dataset
        numClasses = int(f.readline())
        classes = f.readline().rstrip().split(",")
        numAttr = int(f.readline().rstrip())

        for i in range(numAttr):
            line = f.readline().rstrip().split(",")
            factors.extend(line[2:])

            for j in range(int(line[1])):
                attr[line[0]] = line[2:]

        numData = int(f.readline())

        for i in range(numData):
            data[i] = f.readline().rstrip().split(",")

    return numClasses, classes, numAttr, factors, attr, numData, data


def class_counts(data, numClasses, numAttr, numData, classes):
    total_class_counts = {}
    factor_counts = {}

    for label in classes:
        total_class_counts[label] = 0
        factor_counts[label] = {}
        for idx in factors:
            factor_counts[label][idx] = 0

    for i in range(numData):
        for j in range(numClasses):
            if data[i][numAttr] == classes[j]:
                total_class_counts[classes[j]] += 1
                for k in range(numAttr):
                    factor_counts[classes[j]][data[i][k]] += 1

    return total_class_counts, factor_counts


def find_set_entropy(total_class_counts, numClasses, numData):
    set_entropy = 0
    for i in range(numClasses):
        prob = total_class_counts[classes[i]] / numData
        set_entropy -= prob * np.log2(prob)

    return set_entropy


def find_entropy(counts, numClasses, factors):
    entropy = {}

    for idx in factors:
        entr = 0
        numData = 0
        for i in range(numClasses):
            numData += counts[classes[i]][idx]
        for i in range(numClasses):
            prob = counts[classes[i]][idx] / numData
            entr -= prob * np.log2(prob)

        if np.isnan(entr):
            entr = 0
        entropy[idx] = entr

    return entropy


def find_gain(entropy, counts, numData, numClasses, attr, set_entropy):
    gain = {}
    for key in attr.keys():
        g = 0
        for factor in attr[key]:
            num_factor = 0
            for i in range(numClasses):
                num_factor += counts[classes[i]][factor]

            g -= (num_factor / numData) * entropy[factor]

        gain[key] = g + set_entropy

    return gain


def find_root_node():
    set_entropy = find_set_entropy(total_class_counts, numClasses, numData)
    # print(set_entropy)

    entropy = find_entropy(factor_counts, numClasses, factors)
    # print(entropy)

    gain = find_gain(entropy, factor_counts, numData, numClasses, attr, set_entropy)

    return max(gain, key=gain.get)


classes = []
attr = {}
data = {}
factors = []
line = []

if __name__ == "__main__":

    numClasses, classes, numAttr, factors, attr, numData, data = open_data(
        "./fishing_data.txt"
    )
    total_class_counts, factor_counts = class_counts(
        data, numClasses, numAttr, numData, classes
    )
    root = find_root_node()
    print(root)
