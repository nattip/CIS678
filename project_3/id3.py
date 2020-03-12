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


def exclude_iter(iterable, values):
    return [v for v in iterable if v not in values]


def chain_calls(iterable, operation="sub"):
    if operation == "sub":
        return eval("-".join([str(i) for i in iterable]))
    elif operation == "add":
        return eval("+".join([str(i) for i in iterable]))


class AttributeBound:
    def __init__(self, parent, value):
        self.parent = parent
        self.value = value

    def check_bounds(self, row: Tuple[str]):
        return self.value in row

    def __repr__(self):
        return f"Attribute: {self.parent} Value: {self.value}"


class Column:
    def __init__(self, heading, count, values):
        self.heading = heading
        self.count = count
        self.values = values

    def __repr__(self):
        return (
            f"Heading: {self.heading}\n"
            + f"Count: {self.count}\n"
            + f"Values: {self.values}"
        )


class FishingDataLoader:
    def __init__(self, filename):
        with open(filename) as f:
            f.readline()
            self.classes = f.readline().rstrip().split(",")
            n = int(f.readline().rstrip())

            self.columns = []

            for i in range(n):
                line = f.readline().rstrip().split(",")
                heading = line[0]
                count = int(line[1])
                values = line[2:]

                self.columns.append(Column(heading, count, values))

            n = int(f.readline().rstrip())
            self.wind = []
            self.water = []
            self.air = []
            self.forecast = []
            self.label = []

            for i in range(n):
                line = f.readline().rstrip().split(",")
                self.wind.append(line[0])
                self.water.append(line[1])
                self.air.append(line[2])
                self.forecast.append(line[3])
                self.label.append(line[4])

            self.dataset = {
                "wind": self.wind,
                "water": self.water,
                "air": self.air,
                "forecast": self.forecast,
                "label": self.label,
            }
            self.positive_label = "Yes"
            self.negative_label = "No"

    def __len__(self):
        return len(self.dataset["wind"])

    def to_iterator(self):
        return zip(
            self.dataset["wind"],
            self.dataset["water"],
            self.dataset["air"],
            self.dataset["forecast"],
            self.dataset["label"],
        )

    def select(self, bounds: List[AttributeBound] = None):
        if bounds is None:
            return self.dataset

        dataset_copy = copy.deepcopy(self.dataset)
        to_delete = []
        for i in range(len(self)):
            full_row = (
                self.wind[i],
                self.water[i],
                self.air[i],
                self.forecast[i],
            )
            for bound in bounds:
                if not bound.check_bounds(full_row):
                    to_delete.append(i)

        for key in self.dataset.keys():
            dataset_copy[key] = [
                dataset_copy[key][v]
                for v in range(len(dataset_copy[key]))
                if v not in to_delete
            ]

        return dataset_copy


class DecisionTree:
    def __init__(self, filename):
        self.total_positive = 0
        self.total_negative = 0
        self.total_entropy = 0
        self.root_node = ""
        self.tree = nx.Graph()

        self.data = FishingDataLoader(filename)
        self.total_value_count = len(self.data)

        self._set_pos_neg()
        self._set_total_entropy()
        self._set_root_node()

    def __repr__(self):
        return (
            f"nPos: {self.total_positive}\n"
            + f"nNeg: {self.total_negative}\n"
            + f"total entropy: {self.total_entropy}\n"
            + f"root node: {self.root_node}"
        )

    def _get_gain(self, value, total, entropy):
        return (value / total) * entropy

    def _get_pos_neg_counts(self, raw, rows, key):
        # Count pos/neg outcomes for key
        poscount = 0
        negcount = 0
        for i, value in enumerate(rows):
            if value == key and raw["label"][i] == self.data.positive_label:
                poscount += 1
            elif value == key and raw["label"][i] == self.data.negative_label:
                negcount += 1

        return poscount, negcount

    def _get_pos_neg_counts_for_subset(self, raw, rows):
        # Count pos/neg outcomes for key
        poscount = 0
        negcount = 0
        for i, value in enumerate(rows):
            if raw["label"][i] == self.data.positive_label:
                poscount += 1
            elif raw["label"][i] == self.data.negative_label:
                negcount += 1

        return poscount, negcount

    def _get_proportion_and_entropy_for_attr(self, attribute, bounds=None):
        raw = self.data.select(bounds)
        rows = raw[attribute]
        attr_counts = Counter(rows)
        total_counts = sum(attr_counts.values())

        key_pos_neg = {}

        for key in attr_counts.keys():
            poscount, negcount = self._get_pos_neg_counts(raw, rows, key)

            entropy = chain_calls(
                [
                    self._get_entropy(poscount, attr_counts[key]),
                    self._get_entropy(negcount, attr_counts[key]),
                ],
                "add",
            )

            key_pos_neg[key] = {
                "pos": poscount,
                "neg": negcount,
                "entropy": entropy,
                "gain": self._get_gain(poscount + negcount, total_counts, entropy),
            }

        return key_pos_neg

    def _get_entropy(self, value, total):
        if value == 0:
            return 0
        P = value / total
        return -(P * np.log2(P))

    def _find_next_node(
        self, total_entropy, total_positive, total_negative, dataset, boundary=None,
    ):
        gainz = {}

        for attribute in exclude_iter(list(dataset.keys()), "label"):
            proportions = self._get_proportion_and_entropy_for_attr(attribute, boundary)

            gains_to_calc = [total_entropy] + [
                value["gain"] for value in proportions.values()
            ]

            gainz[attribute] = chain_calls(gains_to_calc)

        root_node = max(gainz, key=gainz.get)
        return root_node

    def _set_root_node(self):
        self.root_node = self._find_next_node(
            self.total_entropy,
            self.total_positive,
            self.total_negative,
            self.data.dataset,
        )
        self.tree.add_node(self.root_node)

    def _set_pos_neg(self):
        for row in self.data.dataset["label"]:
            if row == "Yes":
                self.total_positive += 1
            else:
                self.total_negative += 1

    def _set_entropy(self, pos, neg, total):
        return self._get_entropy(pos, total) + self._get_entropy(neg, total)

    def _set_total_entropy(self):
        self.total_entropy = self._set_entropy(
            self.total_positive, self.total_negative, self.total_value_count
        )

    def fit(self, node):
        next_node = None
        routes = list(set(self.data.dataset[node]))

        for route in routes:
            # Replace sunny with route
            for key in exclude_iter(self.data.dataset.keys(), ["label", node]):
                boundary = AttributeBound(node, route)
                data = self.data.select([boundary])
                pos, neg = self._get_pos_neg_counts_for_subset(data, data[key])

                if pos + neg == 1:
                    self.tree.add_edge(node, f"Yes{route}")
                    continue

                entropy_for_set = self._set_entropy(pos, neg, pos + neg)

                next_node = self._find_next_node(
                    entropy_for_set, pos, neg, data, [boundary]
                )
                self.tree.add_edge(node, next_node)

        self.fit(next_node)

    def show(self):
        plt.subplot(121)
        nx.draw(self.tree, with_labels=True, font_weight="bold")
        plt.show()


if __name__ == "__main__":
    d = DecisionTree("./fishing.txt")
    d.fit(d.root_node)
    d.show()


# calculate entropy for each factor of each individual attribute
# calculate gain for each attribute
# find largest gain
# recurse and recalculate from dataset without data from given node
