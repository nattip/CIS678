# Written by Natalie Tipton
# February 2, 20202
# CIS 678 - Machine Learning
# Spam Classification of text messages using
#   the Naive Bayes algorithm

from collections import Counter
import string

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
    total_unique_words = set()

    # Probabilities of words in each class
    prob_ham = []
    prob_spam = []

    # read in file one line at a time
    with open("./data.data") as f:
        lines = f.readlines()

        # Read all of the lines and append them to their classification
        for line in lines:
            if line.startswith("ham"):
                # Get the whole string
                raw_text = "".join(line.split("ham"))[1:].rstrip()
                # split string by spaces and append each word appropriately
                for val in raw_text.split(" "):
                    all_words.append(val)
                    ham_raw.append(val)

            if line.startswith("spam"):
                raw_text = "".join(line.split("spam"))[1:].rstrip()
                for val in raw_text.split(" "):
                    all_words.append(val)
                    spam_raw.append(val)

        # sort words into different places
        cleaned_words = clean_words(all_words)
        cleaned_ham_words = clean_words(ham_raw)
        cleaned_spam_words = clean_words(spam_raw)

        # count up occurrences of each word in all places
        cleaned_words_counts = Counter(cleaned_words)
        cleaned_ham_word_counts = Counter(cleaned_ham_words)
        cleaned_spam_word_counts = Counter(cleaned_spam_words)

        # count number of words in all messages
        total_words_with_duplicates = len(cleaned_words)

        # print(cleaned_spam_words_counts["hi"], cleaned_words_counts["hi"])
        total_unique_words = set(cleaned_words)

        # TODO figure out how to get just values from cleaned_spam/ham_word_counts
        for word in range(len(total_unique_words)):
            prob_spam[0] = (cleaned_spam_word_counts[0] + 1) / (
                len(total_words_with_duplicates) + len(total_unique_words)
            )
            prob_ham[0] = (cleaned_ham_word_counts[0] + 1) / (
                total_words_with_duplicates + len(total_unique_words)
            )

        print(total_words_with_duplicates)
        print(len(total_unique_words))
        print(prob_ham)

        # for uword in total_unique_words:
        #     print(
        #         f"occurance of word: {uword} in "
        #         f"ham: {cleaned_ham_words_counts[uword]} in "
        #         f"spam: {cleaned_spam_words_counts[uword]}"
        #     )

        # Use this to find words that are common in everything
        # and maybe not count thoes for bayes
        # print(cleaned_words.most_common(5))
