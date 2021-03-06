import numpy as np
import mahotas as mh
import os, os.path

def haralick(dirs):
    # Extract texture feature for each image based on Haralick method

    features = []

    for idir in range(len(dirs)):
        files = [name for name in os.listdir(dirs[idir]) 
                 if os.path.isfile(os.path.join(dirs[idir], name))]

        for ifile in range(len(files)):
            if files[ifile][-3:] != 'jpg': # ignore non-image files
                continue
            image = mh.imread(dirs[idir]+files[ifile]) # read image 
            if (len(image.shape) == 3):    # convert to gray if colored
                image = mh.colors.rgb2gray(image, dtype=np.uint8)

            features.append(mh.features.haralick(image).mean(0))

    features = np.array(features)        
    return features
