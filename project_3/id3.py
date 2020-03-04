# Written by Natalie Tipton
# March 10, 20202
# CIS 678 - Machine Learning
# Dr. Greg Wolffe
# ID3 decision tree creator

import pandas as pd
import numpy as np


class DecisionTree:
    def __init__(self, attributes, dataset, classes, groups):
        self.attributes = attributes
        self.classes = classes
        self.dataset = dataset
        self.groups = groups

    def fit(self):
        for row in self.dataset.iterrows():
            pass


if __name__ == "__main__":
    classes = {"count": 0, "values": []}
    attributes = {"count": 0, "values": []}

    # Pandas dataframe to hold the dataset
    dataset = None

    with open("data.txt") as f:
        # Get Output Classes
        # Gets the number of values
        classes["count"] = int(f.readline().rstrip())
        # Strips \n and \r
        classes = f.readline().rstrip().split(",")

        # Get Attributes
        attributes["count"] = int(f.readline().rstrip())

        if attributes["count"] > 1:
            for i in range(attributes["count"]):
                attributes["values"].append(
                    f.readline().rstrip().split(",")[0]
                )

        # Make Dataset
        num_values = int(f.readline().rstrip())
        list_of_values = []

        for i in range(num_values):
            values = f.readline().rstrip().split(",")
            attributes_with_classification = attributes["values"] + ["label"]

            dataframe = {}

            for key, value in zip(attributes_with_classification, values):
                dataframe[key] = value

            list_of_values.append(dataframe)
            dataframe = {}

        dataset = pd.DataFrame(list_of_values)

    groups = {}
    for attr in attributes["values"]:
        groups[attr] = pd.DataFrame(dataset.groupby([attr, "label"]).size())
        groups[attr].reset_index(inplace=True)
    print(groups)

