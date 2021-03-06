import os, os.path
import mahotas as mh
import numpy as np
from mahotas.features import surf
from sklearn.cluster import KMeans

def sobel(dirs):
    # Create acutance feature:

    features = []  # store local feature descriptors

    for idir in range(len(dirs)):
        files = [name for name in os.listdir(dirs[idir]) 
                 if os.path.isfile(os.path.join(dirs[idir], name))]

        for ifile in range(len(files)):
            if files[ifile][-3:] != 'jpg':  # ignore non-image files
                continue
            image = mh.imread(dirs[idir]+files[ifile]).astype(np.uint8) # read image       
            if (len(image.shape) == 3):  # convert to gray if colored
                image = mh.colors.rgb2gray(image, dtype=np.uint8)

            # Calculate the object edges
            edges = mh.sobel(image, just_filter=True)
            edges = edges.ravel()  # flatten the array

            features.append(np.sqrt(np.dot(edges, edges)))

    features = np.array(features) 
    return features
