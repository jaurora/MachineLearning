#!/usr/bin/env python
import pandas as pd
import numpy as np

import stats_feature as sf
import cross_feature as cf

def itera(dcols):
    for key, val in dcols.items():
        print(key, val)

##### load the train file into a dataframes ##### 
df = pd.read_csv('./LoanStats3b.csv', header=1, low_memory=False)  
# delete last two rows
nlines = len(df)
df = df.drop(df.index[[nlines-2, nlines-1]])

##### feature visualization #####

cols  = df.columns.tolist()
dict_cols = {}
for icol in range(len(cols)):
    dict_cols[icol] = cols[icol] 

itera(dict_cols)
scol = input('Feature to Visualize [1-51], [-1]->Exit: ')
while (scol != -1):
    sf.vis_feature(df[cols[scol]])
    scol = input('Feature to Visualize [1-51], [-1]->Exit: ') 


index_train = (df['loan_status'] == 'Fully Paid') | (df['loan_status'] == 'Charged Off')
train_set  = df[index_train]

scol = input('Feature to Couple with Loan Status [1-51], [-1]->Exit: ')
while (scol != -1):
    cf.cross_hist(train_set[cols[scol]], train_set[cols[16]])
    scol = input('Feature to Visualize [1-51], [-1]->Exit: ') 


