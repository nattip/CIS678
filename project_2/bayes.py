# Written by Natalie Tipton
# February 2, 20202
# CIS 678 - Machine Learning
# Dr. Greg Wolfefe
# Spam/Ham Classification of text messages using
#   the Naive Bayes algorithm

from collections import Counter
import string
import math
import heapq
import matplotlib.pyplot as plt
import numpy as np

#################################################
# Function to clean incoming data:
#   Turn all letters lower case
#   Remove all punctuation
#   Remove all numeric-only strings


def clean_words(all_words):

    cleaned_words = []
    for word in all_words:
        word = word.lower()

        for character in word:
            if character in string.punctuation:
                word = word.replace(character, "")

        if not word.isnumeric() and word != "":
            cleaned_words.append(word)

    return cleaned_words


#################################################

if __name__ == "__main__":

    # Raw text sorting
    ham_raw = []
    spam_raw = []

    # Unique words for ham or spam
    all_words = []
    vocab = set()

    # Probabilities of words in each class
    prob_ham = {}
    prob_spam = {}
    prob_diff = {}

    # initial counts for number of ham/spam messages
    ham_count = 0
    spam_count = 0

    # read in file one line at a time
    with open("./train.data") as f:
        lines = f.readlines()

        # Read all of the lines and append them to their classification
        for line in lines:
            if line.startswith("ham"):
                # Get the whole string
                ham_count += 1
                raw_text = "".join(line.split("ham"))[1:].rstrip()
                # split string by spaces and append each word appropriately
                for val in raw_text.split(" "):
                    all_words.append(val)
                    ham_raw.append(val)

            if line.startswith("spam"):
                spam_count += 1
                raw_text = "".join(line.split("spam"))[1:].rstrip()
                for val in raw_text.split(" "):
                    all_words.append(val)
                    spam_raw.append(val)

        # clean words in all classifications
        cleaned_words = clean_words(all_words)
        cleaned_ham_words = clean_words(ham_raw)
        cleaned_spam_words = clean_words(spam_raw)

        # count up occurrences of each word in all places
        cleaned_words_counts = Counter(cleaned_words)
        cleaned_ham_word_counts = Counter(cleaned_ham_words)
        cleaned_spam_word_counts = Counter(cleaned_spam_words)

        # create vocabulary with no repeating words
        vocab = set(cleaned_words)

        # remove 5 most common words in all messages
        vocab.remove("to")
        vocab.remove("i")
        vocab.remove("you")
        vocab.remove("a")
        vocab.remove("the")

        # find sizes of all different sets of words
        vocab_size = len(vocab)
        spam_size = len(cleaned_spam_words)
        ham_size = len(cleaned_ham_words)

        # make dictionaries of spam and ham words with the probability
        # that each word will occur in either classification
        for word in vocab:
            prob_spam[word] = (cleaned_spam_word_counts[word] + 1) / (
                spam_size + vocab_size
            )
            prob_ham[word] = (cleaned_ham_word_counts[word] + 1) / (
                ham_size + vocab_size
            )

            # find the difference between spam and ham probabilities
            # for future analysis of most obvious classifications
            prob_diff[word] = prob_ham[word] - prob_spam[word]

        # Use this to find words that are common in everything
        # and maybe not count those for bayes
        # print(cleaned_words_counts.most_common(5))

        # calculate overall percentage of spam and ham messages in training
        prob_of_ham_message = ham_count / (ham_count + spam_count)
        prob_of_spam_message = spam_count / (ham_count + spam_count)

    # lists of true and hypothesized classifications
    true_class = []
    hyp_class = []

    # open test data line by line
    with open("./test.data") as f:
        lines = f.readlines()

        # set counters for true and hypothesized classes to 0
        test_ham_count = 0
        test_spam_count = 0
        hyp_ham_count = 0
        hyp_spam_count = 0

        # go through lines 1 at a time
        for line in lines:
            # reset the big product for naive bayes calculation back to 0
            big_product_ham = 0
            big_product_spam = 0

            # pull the true classification off the message into a list
            true_class.append(line.split()[0])
            # count up the true occurrences for percentage calculations
            if line.split()[0] == "ham":
                test_ham_count += 1
            if line.split()[0] == "spam":
                test_spam_count += 1

            # leave only the message w/o the classification remaining
            message = line.split()[1:]

            # for each word in the message
            for word in message:
                # do not count if word is not in vocabulary
                if word not in vocab:
                    continue

                # naive bayes formula using log rules
                big_product_ham += math.log10(prob_ham[word])
                big_product_spam += math.log10(prob_spam[word])

            cnb_ham = math.log10(prob_of_ham_message) + big_product_ham
            cnb_spam = math.log10(prob_of_spam_message) + big_product_spam

            # classify message
            if cnb_ham > cnb_spam:
                hyp_class.append("ham")
                hyp_ham_count += 1
            elif cnb_ham < cnb_spam:
                hyp_class.append("spam")
                hyp_spam_count += 1
            else:
                hyp_class.append("ham")
                hyp_ham_count += 1

        true_pos = 0
        true_neg = 0
        false_pos = 0
        false_neg = 0

        # count up true and false pos and negs
        for i in range(0, len(true_class)):
            if true_class[i] == "spam" and hyp_class[i] == "spam":
                true_pos += 1
            elif true_class[i] == "ham" and hyp_class[i] == "ham":
                true_neg += 1
            elif true_class[i] == "spam" and hyp_class[i] == "ham":
                false_neg += 1
            elif true_class[i] == "ham" and hyp_class[i] == "spam":
                false_pos += 1

        # calculate metrics for model
        correct = (
            100 * (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
        )
        recall = 100 * (true_pos / (true_pos + false_neg))
        tnr = 100 * (true_neg / (true_neg + false_pos))
        precision = 100 * (true_pos / (true_pos + false_pos))

        #################################################
        # Output
        #   Printed metrics
        #   Pie Charts
        #   Bar Charts

        print("\ntrue positives =", true_pos)
        print("true negatives =", true_neg)
        print("false positives =", false_pos)
        print("false negatives =", false_neg)
        print("\nRecall =", recall)
        print("Precision =", precision)
        print("True Negative Rate =", tnr)
        print("correct classification =", correct, "\n")

        # determining what the most telling spam and ham words are
        # print(heapq.nlargest(5, prob_diff, key=prob_diff.get))
        # print(heapq.nsmallest(5, prob_diff, key=prob_diff.get))

        # create pie charts to show percentage of spam and ham
        # messages in training and testing and what the model
        # determined for the testing set
        labels = "Spam", "Ham"
        sizes = [prob_of_spam_message, prob_of_ham_message]
        explode = (0.1, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

        fig1, ax1 = plt.subplots()
        ax1.pie(
            sizes,
            explode=explode,
            labels=labels,
            autopct="%1.1f%%",
            shadow=True,
            startangle=90,
        )
        ax1.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.title("True Training Percentages")
        plt.show()

        labels = "Spam", "Ham"
        sizes = [
            test_spam_count / (test_spam_count + test_ham_count),
            test_ham_count / (test_spam_count + test_ham_count),
        ]
        explode = (0.1, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

        fig2, ax1 = plt.subplots()
        ax1.pie(
            sizes,
            explode=explode,
            labels=labels,
            autopct="%1.1f%%",
            shadow=True,
            startangle=90,
        )
        ax1.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.title("True Testing Percentages")
        plt.show()

        labels = "Spam", "Ham"
        sizes = [
            hyp_spam_count / (hyp_spam_count + hyp_ham_count),
            hyp_ham_count / (hyp_spam_count + hyp_ham_count),
        ]
        explode = (0.1, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

        fig3, ax1 = plt.subplots()
        ax1.pie(
            sizes,
            explode=explode,
            labels=labels,
            autopct="%1.1f%%",
            shadow=True,
            startangle=90,
        )

        ax1.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.title("Hypothesized Testing Percentages")
        plt.show()

        # create stacked bar charts to show the probability of spam or ham
        # for the most telling spam words and then the most telling ham words
        plt.figure(4)
        N = 5

        # showing probability for the words determined to be most telling for spam
        top_spam_probs = (
            prob_spam["call"],
            prob_spam["free"],
            prob_spam["txt"],
            prob_spam["claim"],
            prob_spam["your"],
        )
        low_ham_probs = (
            prob_ham["call"],
            prob_ham["free"],
            prob_ham["txt"],
            prob_ham["claim"],
            prob_ham["your"],
        )

        ind = np.arange(N)  # the x locations for the groups
        width = 0.35  # the width of the bars: can also be len(x) sequence

        p1 = plt.bar(ind, low_ham_probs, width)
        p2 = plt.bar(ind, top_spam_probs, width, bottom=low_ham_probs)

        plt.ylabel("Probability of Word")
        plt.title("Probability of Spam or Ham\nFor Most Telling Spam Words")
        plt.xticks(ind, ("call", "free", "txt", "claim", "your"))
        plt.legend((p1[0], p2[0]), ("Ham", "Spam"))

        plt.show()

        plt.figure(5)
        N = 5

        # showing probability for the words determined to be most telling for ham
        low_spam_probs = (
            prob_spam["my"],
            prob_spam["me"],
            prob_spam["in"],
            prob_spam["it"],
            prob_spam["u"],
        )
        top_ham_probs = (
            prob_ham["my"],
            prob_ham["me"],
            prob_ham["in"],
            prob_ham["it"],
            prob_ham["u"],
        )

        ind = np.arange(N)  # the x locations for the groups
        width = 0.35  # the width of the bars: can also be len(x) sequence

        p1 = plt.bar(ind, low_spam_probs, width)
        p2 = plt.bar(ind, top_ham_probs, width, bottom=low_spam_probs)

        plt.ylabel("Probability of Word")
        plt.title("Probability of Spam or Ham\nFor Most Telling Ham Words")
        plt.xticks(ind, ("my", "me", "in", "it", "u"))
        plt.legend((p1[0], p2[0]), ("Spam", "Ham"))

        plt.show()

