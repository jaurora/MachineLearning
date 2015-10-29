####################################################################
# This module is used to classification objects
# based on image characteristic features using logstic regression. 
#
# Prediction score by cross validation was about 86%
#
# The training dataset is part of Caltech 101 dataset
# url: http://www.vision.caltech.edu/Image_Datasets/Caltech101/
####################################################################

import numpy as np
import matplotlib.pyplot as plt
import mahotas as mh
import pandas as pd
import os, os.path
import matplotlib as mpl
mpl.rcParams['interactive'] = True

from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score as auc
from sklearn import linear_model
from sklearn import preprocessing

# Import feature construction modules
import sobel_feature as sof
import haralick_feature as haf
import surf_feature as suf
import binarization_feature as bif

# Classifiy three types of objects
dirs = []
dirs.append('./101_ObjectCategories/Faces/')
dirs.append('./101_ObjectCategories/Motorbikes/')
dirs.append('./101_ObjectCategories/BACKGROUND_Google/')

# Create labels for each type of objects
labels = []
for idir in range(len(dirs)):
    files = [name for name in os.listdir(dirs[idir]) 
             if os.path.isfile(os.path.join(dirs[idir], name))]
    for ifile in range(len(files)):
        if files[ifile][-3:] != 'jpg':
            continue
        labels.append(idir)

labels = np.array(labels)

# Build image features
fetr1 = sof.sobel(dirs)                # acutance based on sobel
fetr1 = preprocessing.scale(fetr1)
fetr1 = fetr1.reshape(len(fetr1),1)

fetr2 = haf.haralick(dirs)             # texture by haralick (best!)
fetr2 = preprocessing.scale(fetr2)

fetr3 = bif.binarization(dirs)         # binarization segment
fetr3 = preprocessing.scale(fetr1)
fetr3 = fetr3.reshape(len(fetr3),1)

fetr4 = np.array(suf.surf_model(dirs), dtype='float64')  # speed up robust feature
fetr4 = preprocessing.scale(fetr4)

# Combine features
features = fetr4
features = np.concatenate((features, fetr2), axis=1)
features = np.concatenate((features, fetr1), axis=1)
features = np.concatenate((features, fetr4), axis=1)

X = features
y = labels

# Build a logistic model
logistic = linear_model.LogisticRegression()

rng = np.random.RandomState(1)
skf = cross_validation.StratifiedKFold(y, n_folds=3, shuffle=True, random_state=rng)

# Cross validation
scores = []
for train_index, test_index in skf:
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf = logistic.fit(X_train, y_train)
    ptrain = clf.predict(X_test)
    scores.append(clf.score(X_test, y_test))

print(np.mean(scores))
