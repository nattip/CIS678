# Written by Natalie Tipton
# March 10, 20202
# CIS 678 - Machine Learning
# Dr. Greg Wolffe
# ID3 decision tree creator

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    targetNames = []
    attributes = {}
    attributeNames = {}

    with open("./fishing_data.txt") as f:
        lines = f.readlines()

        for line in range(0, 2):
            if line == 0:
                numTargets = lines[line]
            elif line == 1:
                for word in lines[line].split(","):
                    targetNames.append(word)
            elif line == 2:
                numAttributes = lines[line]

        for line in range(3, 3 + numAttributes + 1):
            if line == numAttributes + 4:
                numExamples = lines[line]
            else:
                attributes[lines[line].split(",")[0]] == attributeNames[lines[line].split(",")[1]]
                for val in range(2, 2 + attributes[lines[line].split(",")[1]]):


