import numpy as np

f = open('GettysburgAddress.txt', 'r')
nytimes = f.read()

print(nytimes)

words = nytimes.count(' ') + nytimes.count('\n') - nytimes.count('\n\n')
sentences = nytimes.count('. ') + nytimes.count('.\n') + nytimes.count('! ') + nytimes.count('!\n') + nytimes.count('? ') + nytimes.count('?\n')

print('words = ', words)
print('sentences = ', sentences)
