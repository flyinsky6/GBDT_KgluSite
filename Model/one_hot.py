from numpy import argmax
import numpy as np
import csv
import pandas as pd
# define input string
df = pd.read_csv('./GluNALL16_new.csv',header=None)
data = df[1]               ###肽段在第2列，就写1，第三列，就写2

data = data.values.tolist()

alphabet = 'ACDEFGHIKLMNPQRSTVWYUX'
# define a mapping of chars to integers
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))
def one_hot(data,alpabet):
       # integer encode input data
       integer_encoded = [char_to_int[char] for char in data]
       # one hot encode
       onehot_encoded = list()
       for value in integer_encoded:
              letter = [0 for _ in range(len(alphabet))]
              letter[value] = 1
              onehot_encoded.append(letter)
       return onehot_encoded

m = len(data)
dataone_hot = np.zeros((m,33,22))
for i in range(m):
       temp = one_hot(data[i], alphabet)
       temp = np.array(temp)
       dataone_hot[i,:,:]=temp

np.save('one_hot.npy',dataone_hot)
# # Process finished with exit code 0vert encoding
# inverted = int_to_char[argmax(temp[1])]
# print(inverted)

