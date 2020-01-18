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
    # if it has some letter before it would help when you see . . . 
    for letter in range(0, len(text)-1):
        if text[letter] == ('.' or '!' or '?'):
            if text[letter - 1].isalpha():
                if text[letter + 1] == (' ' or '\n'):
                    sentences += 1

    # add one sentence for the last sentence of the file
    sentences += 1
    
    sentences2 = text.count('. ') + text.count('.\n') + text.count('! ') + text.count('!\n') + text.count('? ') + text.count('?\n')
    print('originally', sentences2, 'sentences')

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
f = open('MobyDick.txt', 'r')
text = f.read()
words_moby, sentences_moby, syllables_moby, flesch_moby, grade_moby = flesch_index(text)
print("\nMoby Dick has",  sentences_moby, "sentences,", words_moby, "words, and", syllables_moby, "syllables.")
print("Its Flesch Index is", flesch_moby, "and its reading grade level is", grade_moby)

f = open('NYTimes.txt', 'r')
text = f.read()
words_nyt, sentences_nyt, syllables_nyt, flesch_nyt, grade_nyt = flesch_index(text)
print("\nThe NY Times article has",  sentences_nyt, "sentences,", words_nyt, "words, and", syllables_nyt, "syllables.")
print("Its Flesch Index is", flesch_nyt, "and its reading grade level is", grade_nyt)

f = open('GettysburgAddress.txt', 'r')
text = f.read()
words_gettys, sentences_gettys, syllables_gettys, flesch_gettys, grade_gettys = flesch_index(text)
print("\nThe Gettysburg Address has",  sentences_gettys, "sentences,", words_gettys, "words, and", syllables_gettys, "syllables.")
print("Its Flesch Index is", flesch_gettys, "and its reading grade level is", grade_gettys)

# for simple plotting, combine flesch indexes, grades, and titles into lists
flesch = [flesch_moby, flesch_nyt, flesch_gettys]
grades = [grade_moby, grade_nyt, grade_gettys]
titles = ('Moby Dick', 'NY Times', 'Gettysburg Address')

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
plt.title('Flesch Index of Reading')
plt.xticks(yaxis, titles)
plt.ylabel('Flesch Index')
plt.show()




