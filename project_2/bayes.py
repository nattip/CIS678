from collections import Counter
import string


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


if __name__ == "__main__":
    # Raw text
    ham_raw = []
    spam_raw = []

    # Unique words for ham or spam
    all_words = []
    total_unique_words = set()

    with open("./data.data") as f:
        lines = f.readlines()

        # Read all of the lines and append them to their classification
        for line in lines:
            if line.startswith("ham"):
                # Get the whole string
                raw_text = "".join(line.split("ham"))[1:].rstrip()
                for val in raw_text.split(" "):
                    all_words.append(val)
                    ham_raw.append(val)

            if line.startswith("spam"):
                raw_text = "".join(line.split("spam"))[1:].rstrip()
                for val in raw_text.split(" "):
                    all_words.append(val)
                    spam_raw.append(val)

        cleaned_words = clean_words(all_words)
        cleaned_ham_words = clean_words(ham_raw)
        cleaned_spam_words = clean_words(spam_raw)

        cleaned_words_counts = Counter(cleaned_words)
        cleaned_ham_words_counts = Counter(cleaned_ham_words)
        cleaned_spam_words_counts = Counter(cleaned_spam_words)

        total_words_with_duplicates = len(cleaned_words)

        print(cleaned_spam_words_counts["hi"], cleaned_words_counts["hi"])
        total_unique_words = set(cleaned_words)

        for uword in total_unique_words:
            print(
                f"occurance of word: {uword} in "
                f"ham: {cleaned_ham_words_counts[uword]} in "
                f"spam: {cleaned_spam_words_counts[uword]}"
            )

        # Use this to find words that are common in everything
        # and maybe not count thoes for bayes
        # print(cleaned_words.most_common(5))
