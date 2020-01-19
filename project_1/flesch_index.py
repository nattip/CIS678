#########################
# Title: Project 1 - Flesch Kincaid Index
# Class: CIS 678 - Machine Learning
# Professor: Dr. Wolffe
# Date: January 16, 2020
# Description: This program opens text files containing
#   and calculates the Flesch Kincaid Index and 

import numpy as np
import matplotlib.pyplot as plt

##########################################################################

# determine the number of sentences, words, and syllables in a string
# and calculate the flesch index and resulting reading grade level
def flesch_index(text):

    # new word every time there is a space or a new line
    words = text.count(' ') + text.count('\n') - text.count('\n\n')

    sentences = 0

    # new sentence every time there is a . ! or ? followed by a space or new line
    # with some letter in front of the punctuation 
    for letter in range(0, len(text)-1):
        if text[letter] == '.' or text[letter] == '!' or text[letter] == '?':
            if text[letter - 1].isalpha():
                if text[letter + 1] == ' ' or text[letter + 1] == '\n' or text[letter + 1] == '"':
                    sentences += 1

    # add one sentence for the last sentence of the file
    sentences += 1

    vowels = "aeiouy"

    syllables = 0

    # new syllable every time a letter starts with a vowel or a vowel follows a consonant
    for word in text.split():
        word = word.lower()
        if word[0] in vowels:
            syllables += 1
        for letter in range(1, len(word)):
            if word[letter] in vowels and word[letter - 1] not in vowels:
                syllables += 1
        if word.endswith('e'):
            syllables -= 1

    # calculate flesch-kincaid index and corresponding grade level
    flesch = 206.835 - 84.6 * (syllables / words) - 1.015 * (words / sentences)
    grade = round(0.39 * (words / sentences) + 11.8 * (syllables / words) - 15.59)

    return words, sentences, syllables, flesch, grade

##########################################################################

# open each file and run calculations on it
f = open('CommonSense.txt', 'r')
text = f.read()
words1, sentences1, syllables1, flesch1, grade1 = flesch_index(text)
print("\nCommon Sense has",  sentences1, "sentences,", words1, "words, and", syllables1, "syllables.")
print("Its Flesch Index is", flesch1, "and its reading grade level is", grade1)

f = open('finalpaper.txt', 'r')
text = f.read()
words2, sentences2, syllables2, flesch2, grade2 = flesch_index(text)
print("\nMy paper has",  sentences2, "sentences,", words2, "words, and", syllables2, "syllables.")
print("Its Flesch Index is", flesch2, "and its reading grade level is", grade2)

f = open('TomSawyer.txt', 'r')
text = f.read()
words3, sentences3, syllables3, flesch3, grade3 = flesch_index(text)
print("\nThe Adventures of Tom Sawyer has",  sentences3, "sentences,", words3, "words, and", syllables3, "syllables.")
print("Its Flesch Index is", flesch3, "and its reading grade level is", grade3)

# for simple plotting, combine flesch indexes, grades, and titles into lists
flesch = [flesch1, flesch2, flesch3]
grades = [grade1, grade2, grade3]
titles = ('Common Sense', 'AFib Detection Using Entropy', 'Tom Sawyer')

yaxis = np.arange(len(titles))

# subplots of flesch index and grade levels of each reading
plt.figure(1)

plt.subplot(211)
plt.bar(yaxis, grades)
plt.title('Grade Level of Reading Comprehension')
plt.xticks(yaxis, titles)
plt.ylabel('Reading Grade Level')

plt.subplot(212)
plt.bar(yaxis, flesch)
plt.title('Flesch-Kincaid Index of Reading')
plt.xticks(yaxis, titles)
plt.ylabel('Flesch Index')
plt.show()



