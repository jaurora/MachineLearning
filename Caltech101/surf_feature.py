import os, os.path
import mahotas as mh
import numpy as np
from mahotas.features import surf
from sklearn.cluster import KMeans

def surf_model(dirs):
    # Build local features based on Speeded Up Robust Feature (SURF)

    descriptors = []  # store local feature descriptors

    for idir in range(len(dirs)):
        files = [name for name in os.listdir(dirs[idir]) 
                 if os.path.isfile(os.path.join(dirs[idir], name))]

        for ifile in range(len(files)):
            if files[ifile][-3:] != 'jpg':  # ignore non-image files
                continue
            image = mh.imread(dirs[idir]+files[ifile]).astype(np.uint8) # read image       
            if (len(image.shape) == 3):  # convert to gray if colored
                image = mh.colors.rgb2gray(image, dtype=np.uint8)

            descriptors.append(surf.surf(image, descriptor_only=True))
        
    # Use select one every 32 features from all descriptors
    concat = np.concatenate(descriptors)
    con32 = concat[::32]

    # Cluster descriptors into 256 clusters
    k = 256
    km = KMeans(k)
    km.fit(con32)   # training KMeans model

    # Build the characteristic vector for each image
    # in the clustering space
    features = []
    for d in descriptors:
        cls = km.predict(d)
        tmp = np.array([np.sum(cls == ci) for ci in range(k)])
        features.append(tmp)

    # Return feature vectors
    features = np.array(features) 
    return features 
