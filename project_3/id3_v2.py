# Written by Natalie Tipton
# March 12, 20202
# CIS 678 - Machine Learning
# Dr. Greg Wolffe
# ID3 decision tree creator

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import copy
import pprint


##########################################################################
# read in metadata from data set then all lines of data


def open_data(filename):
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

        # read in actual data
        for i in range(numData):
            data[i] = f.readline().rstrip().split(",")

    return numClasses, classes, numAttr, factors, attr, numData, data


##########################################################################
# count the number of times each classfication label was found
# for the total data set as well as for each individual factor


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


##########################################################################
# Calculate set entropy of entire dataset


def find_set_entropy(total_class_counts, numClasses, numData):
    set_entropy = 0
    for i in range(numClasses):
        prob = total_class_counts[classes[i]] / numData
        set_entropy -= prob * np.log2(prob)

    return set_entropy


##########################################################################
# calculate entropies of each factor in dataset


def find_entropy(counts, numClasses, factors):
    entropy = {}

    # loop through all factors
    for idx in factors:
        entr = 0
        numData = 0

        # count up total occurrences of current factor in the data
        for i in range(numClasses):
            numData += counts[classes[i]][idx]

        # if there are no occurrences, entropy will be 0
        if numData == 0:
            entropy[idx] = 0
            continue
        for i in range(numClasses):
            prob = counts[classes[i]][idx] / numData
            entr -= prob * np.log2(prob)

        # fix for when log2 returns NaN
        if np.isnan(entr):
            entr = 0
        entropy[idx] = entr

    return entropy


##########################################################################
# calculate gains for each attribute in dataset


def find_gain(entropy, counts, numData, numClasses, attr, set_entropy, factors):
    gain = {}

    # loop through all attributes
    for key in attr.keys():
        g = 0
        # loop through the factors for each attribute
        for factor in attr[key]:
            # if factor not in factors:
            #     continue
            num_factor = 0
            # count all occurrences of that factor in the current dataset
            for i in range(numClasses):
                num_factor += counts[classes[i]][factor]

            g -= (num_factor / numData) * entropy[factor]

        gain[key] = g + set_entropy

    return gain


##########################################################################
# find the root node of the tree based on which attribute
# has the highest gain


def find_root_node():
    set_entropy = find_set_entropy(total_class_counts, numClasses, numData)
    entropy = find_entropy(factor_counts, numClasses, factors)
    gain = find_gain(
        entropy, factor_counts, numData, numClasses, attr, set_entropy, factors
    )

    root = max(gain, key=gain.get)
    print("Root node =", root)
    tree[root] = {}
    return root, entropy


##########################################################################
# recursive function that follows down each individual branch
# and finds every node until a final classification can be made


def find_next_node(node, counts, numData, attr, numClasses, entropy, data):

    # loop through the branches off of the root node
    for branch in attr[node]:
        add_to_temp_data = []
        temp_data = {}
        temp_factors = []
        bounds = []
        tree[root][branch] = {}

        # list of the factors that a data point needs
        # to be included down the branch
        bounds.append(branch)

        # loop through all data to find which data points
        # satisfy the location in the tree and add them
        # to the temporary dataset
        for i in range(numData):
            for j in range(numAttr):
                if data[i][j] in bounds:
                    add_to_temp_data.append(i)

        for temp in range(len(add_to_temp_data)):
            temp_data[temp] = data[add_to_temp_data[temp]]

        temp_factors = [e for e in factors if e not in attr[node]]

        temp_attr = copy.deepcopy(attr)
        del temp_attr[node]

        # find updated counts for each classificaiton
        temp_total_counts, temp_factor_counts = class_counts(
            temp_data, numClasses, numAttr, len(add_to_temp_data), classes
        )

        zeros = 0
        for label in classes:
            # if data with a certain factor is never labelled
            # one of the possible classifications, count it
            if temp_factor_counts[label][branch] == 0:
                zeros += 1
            else:
                pos_factor = label

        # if a factor only exist as one classification, end node
        # with classification label
        if zeros == (numClasses - 1):
            next_node = pos_factor
            print("Next node =", next_node)
            if not tree[root][branch]:
                # print(tree[root][branch])
                tree[root][branch] = next_node
            else:
                # print(tree[root][branch])
                tree[root][branch + "1"] = next_node
            # tree[root][branch] = next_node

            continue

        # determine set entropy, factor entropies, and gain
        set_entropy = entropy[branch]
        temp_entropy = find_entropy(temp_factor_counts, numClasses, temp_factors)
        temp_gain = find_gain(
            temp_entropy,
            temp_factor_counts,
            len(add_to_temp_data),
            numClasses,
            temp_attr,
            set_entropy,
            temp_factors,
        )

        # decide next node
        next_node = max(temp_gain, key=temp_gain.get)
        print("Next node =", next_node)

        if not tree[root][branch]:
            # print(tree[root][branch])
            tree[root][branch] = next_node
        else:
            # print(tree[root][branch])
            tree[root][branch + "1"] = next_node

        print(tree)
        # call function again to find next node given
        # current information
        find_next_node(
            next_node,
            temp_factor_counts,
            len(add_to_temp_data),
            temp_attr,
            numClasses,
            temp_entropy,
            temp_data,
        )


##########################################################################
##########################################################################

# global structures
classes = []
attr = {}
data = {}
factors = []
line = []
tree = {}

if __name__ == "__main__":
    # open and read data
    numClasses, classes, numAttr, factors, attr, numData, data = open_data(
        "./fishing_data.txt"
    )

    # count up intial occurrences of each class
    total_class_counts, factor_counts = class_counts(
        data, numClasses, numAttr, numData, classes
    )

    # find the first node of the tree
    root, entropy = find_root_node()

    # build entire tree
    find_next_node(root, factor_counts, numData, attr, numClasses, entropy, data)

    print(tree)

