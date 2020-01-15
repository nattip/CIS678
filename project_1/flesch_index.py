import numpy as np

# gettysburg, ny times, moby dick (lowest to highest)
# turn number into grade level maybe

f = open('MobyDick.txt', 'r')
nytimes = f.read()

print(nytimes)

words = nytimes.count(' ') + nytimes.count('\n') - nytimes.count('\n\n')

#if it has some letter before it would help when you see . . . 
sentences = nytimes.count('. ') + nytimes.count('.\n') + nytimes.count('! ') + nytimes.count('!\n') + nytimes.count('? ') + nytimes.count('?\n')

vowels = "aeiouy"

syllables = 0

for word in nytimes.split():
    word = word.lower()
    if word[0] in vowels:
        syllables += 1
    for letter in range(1, len(word)):
        if word[letter] in vowels and word[letter - 1] not in vowels:
            syllables += 1
    if word.endswith('e'):
        syllables -= 1

flesch = 206.835 - 84.6 * (syllables / words) - 1.015 * (words / sentences)
grade = round(0.39 * (words / sentences) + 11.8 * (syllables / words) - 15.59)
    

print('words = ', words)
print('sentences = ', sentences)
print('syllables = ', syllables)
print('flesch index = ', flesch)
print('grade level = ', grade)
