#!/usr/bin/env python
import sys
import csv
import subprocess
import urllib
import pandas as pd
import numpy as np

# download for url
url = 'https://resources.lendingclub.com/LoanStats3b.csv.zip'
urllib.urlretrieve(url, "LoanStats3b.zip")

# unzip in shell
subprocess.call("unzip LoanStats3b.zip", shell=True)

# read in dataframe
df = pd.read_csv('./LoanStats3b.csv', header=1, low_memory=False)       
# delete last two rows
nlines = len(df)
df = df.drop(df.index[[nlines-2, nlines-1]])

if df.shape[0] < 1000:
    sys.stdout.write('record less than 1000!')
    sys.exit(0)

# random sampling
random_seeds = np.random.choice(len(df)-1, 1000, replace=False)
random_id  = df['id'][random_seeds].values
random_ls  = df['loan_status'][random_seeds].values

# save data to csv
savefile = open("random.csv", "w")
open_file_object = csv.writer(savefile)
open_file_object.writerow(['id','loan_status'])
open_file_object.writerows(zip(random_id, random_ls))
savefile.close()

# print number of lines in csv to screen
sys.stdout.write('Number of lines in random.csv:\n')
sys.stdout.write(str(len(random_id))+'\n')
