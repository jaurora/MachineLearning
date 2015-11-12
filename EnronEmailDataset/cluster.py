#!/usr/bin/python

import math
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.utils import shuffle
from sklearn.cluster import KMeans

feature_vector = np.loadtxt('feature_vector.txt')

with open('usrs.pkl', 'rb') as handle:
  usrs = pickle.load(handle)

with open('word_vector.pkl', 'rb') as handle:
  word_vector = pickle.load(handle)


##### cluster data into K=1..20 clusters #####

n_clusters = [i for i in range(2,11,1)]
inertia = []

for icluster in n_clusters:
    print(icluster)
    clusterer = KMeans(n_clusters=icluster, random_state=10, n_jobs=1)
    clusterer.fit_predict(feature_vector)
    centroids = clusterer.cluster_centers_
    inertia.append(clusterer.inertia_)
    labels = clusterer.labels_

#### Elbow curve for kmeans ####
plt.figure()
plt.plot(n_clusters, inertia, 'b*-')
plt.xlabel('Number of clusters')
plt.ylabel('Sum of distances of samples to their centroids')
plt.title('Evolution for KMeans clustering')
plt.grid(True)
plt.savefig('elbow.png')
plt.close()

usrs = np.array(usrs)
labels = np.array(labels)
word_vector = np.array(word_vector)
K = 10
clus_words = []
clus_usrs = []
clus_ratio = []
for ik in range(K):
    center = centroids[ik]
    index = sorted(range(len(center)), key=lambda x: center[x])
    index = index[::-1]
    center = center[index]
    wvect = word_vector[index]

    clus_words.append(wvect[:10])
    clus_ratio.append(center[:10])
    clus_usrs.append(usrs[labels == ik])
    

#### Plot group information for group 7 ####
plt.figure()
index = np.arange(10)
bar_width = 0.3
opacity = 0.4
plt.bar(index, clus_ratio[1][:10]*1.e2, alpha=opacity, color='b', label='Group 7')
plt.xticks(index + bar_width, clus_words[1][:10])
plt.xlabel('Top 10 Words')
plt.ylabel('Word Frequency [10' +r'$^{-2}$'+']')
plt.title('Word Frequency of Group 2')
plt.grid(True)
plt.savefig('wordfrequency.png')



