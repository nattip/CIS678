# Written by Natalie Tipton
# March 10, 20202
# CIS 678 - Machine Learning
# Dr. Greg Wolfefe
# ID3 decision tree creator

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    targetNames = []
    attributeNames = []

    with open("./fishing_data.txt") as f:
        lines = f.readlines()

        for line in range (0,6):
            if line == 0:
                numTargets = lines[line]
            elif line == 1:
                targetNames = lines[line].split("")
            elif line == 2:
                numAttributes = lines[line]
            elif line == 3:
                for word in lines[line]:
                    
