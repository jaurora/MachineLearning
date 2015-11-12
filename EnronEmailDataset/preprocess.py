import os
import re
import math
import numpy as np
import nltk
import operator
import pickle
from pattern.en import tag

def getwords(doc, stopwords):
    splitter = re.compile('\\W*')
    words = [s.lower() for s in splitter.split(doc) if len(s)>2 and len(s)<20]
    words = [w for w in words if not w in stopwords and not any(i.isdigit() for i in w)]
    #words = nltk.word_tokenize(doc)
   
    ####Normalize with a word list
    #wnl = nltk.WordNetLemmatizer()
    #words = [wnl.lemmatize(word) for word in words]
    
    ####Tag words and select nouns
    #words = [w for w in words if tag(w)[0][1] == u'NN']
   
    #### Stemmer words
    stemmer = nltk.stem.SnowballStemmer("english")
    words = [str(stemmer.stem(word)) for word in words]

    # Tag words
    #tagged = nltk.pos_tag(words)
    #noun = [word for (word, tag) in tagged if tag=='NN']
    #verb = [word for (word, tag) in tagged if tag=='VB']

    words = np.array(words)
    #print(words)
    return dict([(str(w),len(words[words==w])) for w in words])


def readin(readir, stopwords):
    ignore = ['_sent_mail', '_sent', 'sent', 'sent_items', 'contacts','deleted_items', 'all_documents', 'discussion_threads', 'personal']
    subdir = os.listdir(readir)
    output = []
    for idir in subdir:
        if os.path.isfile(os.path.join(readir, idir)):
            fo = open(os.path.join(readir,idir), 'r')
            tmptxt = fo.readline()
            while not ('X-FileName' in tmptxt):
                tmptxt = fo.readline()

            # Ignore fowarded message
            text = []
            for line in fo.readlines():
                if 'Original Message' in line: 
                    break
                text.append(line)
            fo.close()
            x = getwords(''.join(text), stopwords)
            output.append(x)
        else:
            if idir in ignore:
                continue
            output += readin(os.path.join(readir, idir), stopwords)
    return output
