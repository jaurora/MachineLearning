import os, os.path
import mahotas as mh
import numpy as np
from mahotas.features import surf
from sklearn.cluster import KMeans

def binarization(dirs):
    # Create binarization feature:
    # if pixel intensity greater than a threshold, assign it as 1, otherwise 0
    # This feature robust to light

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

            # Calculate the binarization threshold using otsu method
            threshold = mh.thresholding.otsu(image)
            binarized = image > threshold

            # Calculate the ratio between area white and area black           
            nPix = float(image.shape[0]*image.shape[1])
            area_ratio = len(binarized[binarized == 1])/nPix

            features.append(area_ratio)

    features = np.array(features) 
    return features
