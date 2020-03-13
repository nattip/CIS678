# Written by Natalie Tipton
# March 10, 20202
# CIS 678 - Machine Learning
# Dr. Greg Wolffe
# ID3 decision tree creator

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import copy
from networkx import drawing


def open_data(filename):
    # open dataset
    with open(filename) as f:

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
        if numData == 0:
            entropy[idx] = 0
            continue
        for i in range(numClasses):
            prob = counts[classes[i]][idx] / numData
            entr -= prob * np.log2(prob)

        if np.isnan(entr):
            entr = 0
        entropy[idx] = entr

    return entropy


def find_gain(entropy, counts, numData, numClasses, attr, set_entropy, factors):
    gain = {}
    for key in attr.keys():
        g = 0
        for factor in attr[key]:
            if factor not in factors:
                continue
            num_factor = 0
            for i in range(numClasses):
                num_factor += counts[classes[i]][factor]

            g -= (num_factor / numData) * entropy[factor]

        gain[key] = g + set_entropy

    return gain


def find_root_node():
    set_entropy = find_set_entropy(total_class_counts, numClasses, numData)
    entropy = find_entropy(factor_counts, numClasses, factors)
    gain = find_gain(
        entropy, factor_counts, numData, numClasses, attr, set_entropy, factors
    )

    root = max(gain, key=gain.get)
    tree[root] = attr[root]
    print("Root node =", root)

    return root, entropy


def find_next_node(node, counts, numData, attr, numClasses, entropy, data):

    for branch in attr[node]:
        add_to_temp_data = []
        temp_data = {}
        bounds = []
        temp_factors = []
        # print(branch)

        bounds.append(branch)
        for i in range(numData):
            for j in range(numAttr):
                if data[i][j] in bounds:
                    add_to_temp_data.append(i)

        for temp in range(len(add_to_temp_data)):
            temp_data[temp] = data[add_to_temp_data[temp]]

        # print(temp_data)
        temp_factors = [e for e in factors if e not in attr[node]]

        temp_attr = copy.deepcopy(attr)
        del temp_attr[node]

        temp_total_counts, temp_factor_counts = class_counts(
            temp_data, numClasses, numAttr, len(add_to_temp_data), classes
        )

        zeros = 0

        for label in classes:
            if temp_factor_counts[label][branch] == 0:
                zeros += 1
            else:
                pos_factor = label

        if zeros == (numClasses - 1):
            print("Next node =", pos_factor)
            continue

        # print(len(add_to_temp_data))
        # print(temp_factor_counts)
        set_entropy = entropy[branch]
        temp_entropy = find_entropy(temp_factor_counts, numClasses, temp_factors)

        # print("entropy =", temp_entropy)

        temp_gain = find_gain(
            temp_entropy,
            temp_factor_counts,
            len(add_to_temp_data),
            numClasses,
            temp_attr,
            set_entropy,
            temp_factors,
        )

        next_node = max(temp_gain, key=temp_gain.get)
        print("Next node =", next_node)

        find_next_node(
            next_node,
            temp_factor_counts,
            len(add_to_temp_data),
            temp_attr,
            numClasses,
            temp_entropy,
            temp_data,
        )

        # root = max(gain, key=gain.get)


classes = []
attr = {}
data = {}
factors = []
line = []
tree = {}

if __name__ == "__main__":

    numClasses, classes, numAttr, factors, attr, numData, data = open_data(
        "./contact_data.txt"
    )
    total_class_counts, factor_counts = class_counts(
        data, numClasses, numAttr, numData, classes
    )
    root, entropy = find_root_node()

    find_next_node(root, factor_counts, numData, attr, numClasses, entropy, data)

