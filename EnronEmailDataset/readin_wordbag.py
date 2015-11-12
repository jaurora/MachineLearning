#!/usr/bin/python

import os
import re
import math
import numpy as np
import nltk
import operator
import pickle
from pattern.en import tag
import preprocess as ppr
####### read in stop words ########
fo = open('./stopwords.txt')
splitter = re.compile('\\W*')
stopwords = splitter.split(fo.read())

####### data directory ###########
datadir = './maildir/'
usrdirs = os.listdir(datadir)
usrs = []
usrsmail = []

####### extract a word bag from each email for a user ########
for idir in usrdirs:
    print(idir)
    tmpdir = os.path.join(datadir, idir)
    usrsmail.append(ppr.readin(tmpdir, stopwords))
    usrs.append(idir)

####### summarize the word (counts) for each user #########
wordtot = []

for iuser in range(len(usrs)):
    tmpusr = usrsmail[iuser]
    usrbag = {}
    for imail in range(len(tmpusr)):
        mail = tmpusr[imail]
        for word in mail:
            if word in usrbag: usrbag[word] += mail[word]
            else: usrbag[word] = mail[word]        
    wordtot.append(usrbag)

##### save word bag to file
with open('wordtot.pkl', 'wb') as handle:
  pickle.dump(wordtot, handle)

with open('usrs.pkl', 'wb') as handle:
  pickle.dump(usrs, handle)

with open('usrsmail.pkl', 'wb') as handle:
  pickle.dump(usrsmail, handle)
