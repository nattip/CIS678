# Written by Natalie Tipton
# March 10, 20202
# CIS 678 - Machine Learning
# Dr. Greg Wolffe
# ID3 decision tree creator

import pandas as pd
import numpy as np
import pprint


class DecisionTree:
    def __init__(self, attributes, dataset, classes, groups):
        self.attributes = attributes
        self.classes = classes
        self.dataset = dataset
        self.groups = groups
        self.summed_values = self.grouping()

    def _reindex(self, dicty_boi):
        return {v: k for k, v in dicty_boi.items()}

    def grouping(self):
        summed_values = {k: {} for k in self.groups.keys()}
        sums = []
        for attr, df in self.groups.items():
            change_indexes = []
            hits = []
            for index, row in df.iterrows():
                if row[attr] not in hits:
                    hits.append(row[attr])
                    change_indexes.append(index)

            for index, group in enumerate(change_indexes):

                if index + 1 < len(change_indexes):
                    sums.append(
                        (
                            attr,
                            hits[index],
                            self._reindex(
                                df.iloc[group : change_indexes[index + 1]]
                                .set_index(0,)["label"]
                                .to_dict(),
                            ),
                            df.iloc[group : change_indexes[index + 1]][0].sum(),
                        )
                    )
                else:
                    sums.append(
                        (
                            attr,
                            hits[index],
                            self._reindex(
                                df.iloc[group:].set_index(0)["label"].to_dict()
                            ),
                            df.iloc[group:][0].sum(),
                        )
                    )
        for sub in sums:
            sub[2].update({"total": sub[3]})

            summed_values[sub[0]].update({sub[1]: sub[2]})

        return summed_values

        # pp = pprint.PrettyPrinter(indent=4)
        # pp.pprint(summed_values)

    def fit(self):
        entropy = {}
        gain = {}
        flag = 0
        for attr in self.summed_values:
            entropy[attr] = {}
            gain[attr] = 0
            for factor in range(0, len(self.summed_values[attr])):

                if flag == 0:
                    ent_set = -(
                        (
                            self.summed_values[attr][factor]["No"]
                            + self.summed_values[attr][factor + 1]["No"]
                        )
                        / num_values
                        * np.log2(
                            (
                                self.summed_values[attr][factor]["No"]
                                + self.summed_values[attr][factor + 1]["No"]
                            )
                            / num_values
                        )
                        + (
                            self.summed_values[attr][factor]["Yes"]
                            + self.summed_values[attr][factor + 1]["Yes"]
                        )
                        / num_values
                        * np.log2(
                            (
                                self.summed_values[attr][factor]["Yes"]
                                + self.summed_values[attr][factor + 1]["Yes"]
                            )
                            / num_values
                        )
                    )

                flag += 1

                if self.summed_values[attr][factor].get("No") and self.summed_values[
                    attr
                ][factor].get("Yes"):
                    entropy[attr][factor] = -(
                        (
                            self.summed_values[attr][factor]["No"]
                            / self.summed_values[attr][factor]["total"]
                        )
                        * np.log2(
                            self.summed_values[attr][factor]["No"]
                            / self.summed_values[attr][factor]["total"]
                        )
                        + (
                            self.summed_values[attr][factor]["Yes"]
                            / self.summed_values[attr][factor]["total"]
                        )
                        * np.log2(
                            self.summed_values[attr][factor]["Yes"]
                            / self.summed_values[attr][factor]["total"]
                        )
                    )

                elif self.summed_values[attr][factor].get(
                    "No"
                ) and not self.summed_values[attr][factor].get("Yes"):
                    entropy[attr][factor] = -(
                        (
                            self.summed_values[attr][factor]["No"]
                            / self.summed_values[attr][factor]["total"]
                        )
                        * np.log2(
                            self.summed_values[attr][factor]["No"]
                            / self.summed_values[attr][factor]["total"]
                        )
                    )

                elif self.summed_values[attr][factor].get(
                    "Yes"
                ) and not self.summed_values[attr][factor].get("No"):
                    entropy[attr][factor] = -(
                        (
                            self.summed_values[attr][factor]["Yes"]
                            / self.summed_values[attr][factor]["total"]
                        )
                        * np.log2(
                            self.summed_values[attr][factor]["Yes"]
                            / self.summed_values[attr][factor]["total"]
                        )
                    )

                gain[attr] = gain[attr] + (
                    (self.summed_values[attr][factor]["total"] / num_values)
                    * entropy[attr][factor]
                )

            gain[attr] = ent_set - gain[attr]

        print("Entropy:", entropy)
        print("Gain:", gain)
        print("Set entropy:", ent_set)

        # gain_air = ent_set - (
        #     (self.summed_values["Air"]["Warm"]["total"] / num_values) * ent_air_warm
        #     + (self.summed_values["Air"]["Cool"]["total"] / num_values) * ent_air_cool
        # )

        # gain_water = ent_set - (
        #     (self.summed_values["Water"]["Cold"]["total"] / num_values) * ent_water_cold
        #     + (self.summed_values["Water"]["Moderate"]["total"] / num_values)
        #     * ent_water_mod
        #     + (self.summed_values["Water"]["Warm"]["total"] / num_values)
        #     * ent_water_warm
        # )

        # gain_wind = ent_set - (
        #     (self.summed_values["Wind"]["Strong"]["total"] / num_values)
        #     * ent_wind_strong
        #     + (self.summed_values["Wind"]["Weak"]["total"] / num_values) * ent_wind_weak
        # )

        # print("gains")
        # print(gain_air)
        # print(gain_fc)
        # print(gain_water)
        # print(gain_wind)


if __name__ == "__main__":
    classes = {"count": 0, "values": []}
    attributes = {"count": 0, "values": []}

    # Pandas dataframe to hold the dataset
    dataset = None

    with open("fishing_data.txt") as f:
        # Get Output Classes
        # Gets the number of values
        classes["count"] = int(f.readline().rstrip())
        # Strips \n and \r
        classes = f.readline().rstrip().split(",")

        # Get Attributes
        attributes["count"] = int(f.readline().rstrip())

        if attributes["count"] > 1:
            for i in range(attributes["count"]):
                attributes["values"].append(f.readline().rstrip().split(",")[0])

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
    # print(dataset)

    d = DecisionTree(attributes, dataset, classes, groups)

    d.fit()


# calculate entropy for each factor of each individual attribute
# calculate gain for each attribute
# find largest gain
# recurse and recalculate from dataset without data from given node

