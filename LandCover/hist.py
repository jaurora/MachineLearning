import numpy as np
import os
from numpy import genfromtxt
import matplotlib.pyplot as plt

dirs = './images/'
files = os.listdir(dirs)
f1_mean = []
f2_medi = []
f3_diff = []
f4_ratio = []

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



plt.hist(f1_mean)
plt.title('WDRVI Mean Distribution')
plt.savefig('hist_mean.png')
plt.close()

plt.hist(f2_medi)
plt.title('WDRVI Median Distribution')
plt.savefig('hist_median.png')
plt.close()

plt.hist(f3_diff)
plt.title('WDRVI DIFF Distribution')
plt.savefig('hist_diff.png')
plt.close()

plt.hist(f4_ratio)
plt.title('WDRVI Ratio Distribution')
plt.savefig('hist_ratio.png')
plt.close()
