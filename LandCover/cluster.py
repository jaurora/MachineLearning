##########################################################################################
# This module is used to seperate the WDRVI-based land images into two clusters:
# 1 - vegetation 2- soil
##########################################################################################

#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import mahotas as mh
from numpy import genfromtxt
import os

import matplotlib as mpl
mpl.rcParams['interactive'] = True

from sklearn.cluster import KMeans
from sklearn import preprocessing

def cluster():
    choose = input('Basic feature only [1]\nTexture Feature Only [2]\nCombination Feature [3]\nChoose: ')
    dirs = './images/'
    files = os.listdir(dirs)

    f1_mean = []
    f2_medi = []
    f3_diff = []
    f4_ratio = []
    f5_texture = []

    for ifile in files:
        print(ifile)
        if (ifile[-3:] != 'txt'):
            continue
        image = genfromtxt(dirs+ifile, delimiter=',')
    
        select_nan = np.isnan(image)
        data = image[~select_nan]

        f1_mean.append(data.mean())
        f2_medi.append(np.median(data))

        # Sort the wdrvi index 
        data.sort()
        nPix = len(data)
        # Difference between the highest 20% and the lowest 10% data
        f3_diff.append(data[int(nPix*0.8)] - data[int(nPix*0.1)])

        # Define threshold for soil and vegatation based on reference
        above = np.where(data > -0.6)      # need adjustment?
        f4_ratio.append(len(data[above])/float(nPix))

        # Add haralick vector
        image_scale = image/0.8*120+120
        nan_area = np.where(np.isnan(image_scale))
        image_scale[nan_area] = 0
        image_scale = image_scale.astype(int)
        f5_texture.append(mh.features.haralick(image_scale).mean(0))

    f1_mean = np.array(f1_mean).reshape((len(f1_mean), 1))
    f2_medi = np.array(f2_medi).reshape((len(f1_mean), 1))
    f3_diff = np.array(f3_diff).reshape((len(f1_mean), 1))
    f4_ratio = np.array(f4_ratio).reshape((len(f1_mean), 1))
    f5_texture = np.array(f5_texture)

    if (choose == 1):
        features = np.concatenate((f1_mean, f2_medi), axis=1)
        features = np.concatenate((features, f3_diff), axis=1)
        features = np.concatenate((features, f4_ratio), axis=1)
    elif choose == 2:
        features = f5_texture
    else:
        features = np.concatenate((f1_mean, f2_medi), axis=1)
        features = np.concatenate((features, f3_diff), axis=1)
        features = np.concatenate((features, f4_ratio), axis=1)
        features = np.concatenate((features, f5_texture), axis=1)        
        
    # Scale the features
    features = preprocessing.scale(features)

    # Build a clustering model
    # Cluster descriptors into 256 clusters
    k = 2   # two kinds of ground
    km = KMeans(k)
    km.fit(features)   # training KMeans model

    labels = km.predict(features)

    if choose == 1:
        filename = 'labels_basic.txt'
    elif choose == 2:
        filename = 'labels_texture.txt'
        # transform labels for the texture labels
        # labels = (labels+1) % 2
    else:
        filename = 'labels_combo.txt'

    f = open(filename, 'w')
    f.write('File    Label\n')
    for i in range(1, len(files)):
        f.write(files[i] + '  ' + str(labels[i-1]) + '\n')
    f.close()

    # Calcuate the percentage of image of vegatation
    print('Percentage of images of vagetation: ')
    print(len(labels[labels==1])/float(len(labels)))
