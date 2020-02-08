# Written by Natalie Tipton
# February 2, 20202
# CIS 678 - Machine Learning
# Spam Classification of text messages using
#   the Naive Bayes algorithm

from collections import Counter
import string
import math

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

        # Use this to find words that are common in everything
        # and maybe not count thoes for bayes
        # print(cleaned_words_counts.most_common(5))

        prob_of_ham_message = ham_count / (ham_count + spam_count)
        prob_of_spam_message = spam_count / (ham_count + spam_count)

    # lists of true and hypothesized classifications
    true_class = []
    hyp_class = []

    # open test data line by line
    with open("./test.data") as f:
        lines = f.readlines()

        # go through lines 1 at a time
        for line in lines:
            big_product_ham = 0
            big_product_spam = 0

            # pull the true classification off the message into a list
            true_class.append(line.split()[0])
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
            elif cnb_ham < cnb_spam:
                hyp_class.append("spam")
            else:
                hyp_class.append("ham")

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
        sensitivity = 100 * (true_pos / (true_pos + false_neg))
        specificity = 100 * (true_neg / (true_neg + false_pos))

        print("\ntrue positives =", true_pos)
        print("true negatives =", true_neg)
        print("false positives =", false_pos)
        print("false negatives =", false_neg)
        print("\nsensitivity =", sensitivity)
        print("specificity =", specificity)
        print("correct classification =", correct, "\n")
