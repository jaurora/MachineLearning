#!/usr/bin/env python

import pandas as pd
import numpy as np
from patsy import dmatrices, dmatrix
import matplotlib as mpl
mpl.rcParams['interactive'] = True
import xgboost as xgb

import vis_model as vm
import preprocess_data as ppd
import learning as ln

##### load the train file into a dataframes ##### 
df = pd.read_csv('./LoanStats3b.csv', header=1, low_memory=False)       
# delete last two rows
nlines = len(df)
df = df.drop(df.index[[nlines-2, nlines-1]])

# preprocess features
ppd.clean(df)

# save loan status before maping
loansts = df['loan_status']

# maping loan status
maps = {}
status = df['loan_status'].unique()
for item in range(len(status)):
    if (status[item] == 'Fully Paid'):
        maps[status[item]] = 1
    elif (status[item] == 'Charged Off'):
        maps[status[item]] = 0
    else:
        maps[status[item]] = -1 

df['loan_status'] = df['loan_status'].map(lambda x: maps[x])

#### feature selection ####
X_feature = df.columns.tolist()

# removed features list
rm_list = ['loan_status', 'id', 'member_id', 'url','next_pymnt_d', 
           'last_pymnt_d', 'issue_d', 'last_credit_pull_d',
           'total_pymnt', 'total_pymnt_inv',
           'total_rec_prncp','total_rec_int',
           'recoveries', 'collection_recovery_fee',
           'last_pymnt_amnt', 'pymnt_plan']

# categorical feature list
enlist = ['term', 'grade', 'emp_length',  'emp_title', 'desc', 'title', 
          'purpose', 'home_ownership', 'verification_status', 'zip_code',
          'pymnt_plan', 'addr_state', 'initial_list_status', 'subgrade']
    
#### process categorical features ####
strf = ''
for ix in range(len(X_feature)):
    tmp = str(X_feature[ix])
    if X_feature[ix]  in rm_list:
        continue
    if X_feature[ix] in enlist:
        tmp = 'C(' + tmp + ')'
    strf = strf + ' + ' + tmp if ix >  2 else tmp

# create dataframes with an intercept column and dummy variables for occupation and occupation_husb
ty, tX = dmatrices('loan_status ~ ' + strf, df, return_type="dataframe")

# separate training/test sets
select_index = ty['loan_status'] != -1
 
# training set
y = ty['loan_status'][select_index].values
X = tX[select_index].values

# test set
testX = tX[~select_index].values
# test id
testid = df[['member_id']][~select_index]
# current loan status of test set
test_sts = loansts[~select_index].values


### training models ##### 
depth = 9
eta = 0.05
rounds = 100 
trs, tes, bst = ln.learning(X, y, depth, eta, rounds, 1.0)

print(trs)
print(tes)

dtestX = xgb.DMatrix(testX)
output = bst.predict(dtestX)

#### output results ##### 
testid['loan_status'] = output
testid['loan_status'] = testid['loan_status'].map(lambda x: 'Fully Paid' if x>0.5 else 'Charged Off')
testid.to_csv('prediction.csv')


####### visualization ######

#### fully Paid ratio predicted by loan_status class ####
cpdf = testid.copy()
cpdf['cur_loan_status'] = test_sts
vm.fp_ratio(cpdf)

#### plot feature importance #####
vm.feature_importance(X, y, tX.columns)









