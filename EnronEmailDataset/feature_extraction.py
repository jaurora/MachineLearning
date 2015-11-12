#!/usr/bin/python

import math
import numpy as np
import operator
import pickle

#######################################
# Extract a word vector for every user
######################################

# load datasets 
with open('wordtot.pkl', 'rb') as handle:
  wordtot = pickle.load(handle)

with open('usrs.pkl', 'rb') as handle:
  usrs = pickle.load(handle)

# use words with top ?% frequency for each user
# combine them into a feature vector
word_vector = []

for iusr in range(len(usrs)):
    print(iusr)
    bag = wordtot[iusr]
    sortbag = sorted(bag.items(), key=operator.itemgetter(1))[::-1] 
    
    limit = int(0.2*len(sortbag))
    count = 0
    for iword in sortbag:
        if count > limit: break
        if not iword[0] in word_vector:
            word_vector.append(iword[0])
            count += 1

feature_vector = np.zeros((len(usrs), len(word_vector)))
for iusr in range(len(usrs)):
    count = 0
    for iword in range(len(word_vector)):
        if not word_vector[iword] in wordtot[iusr]: continue
        feature_vector[iusr,iword] = wordtot[iusr][word_vector[iword]]
        count += wordtot[iusr][word_vector[iword]]        
    feature_vector[iusr,:] /= np.max([count,1]) 


# save data to file
np.savetxt('feature_vector.txt', feature_vector)

with open('word_vector.pkl', 'wb') as handle:
  pickle.dump(word_vector, handle)
