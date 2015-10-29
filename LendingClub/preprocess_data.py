#!/usr/bin/env python

import pandas as pd
import numpy as np
from collections import Counter
import datetime as dt
import re

##### Load the train file into a dataframes ##### 
def clean(df):
    # column name
    cols = df.columns.tolist()
    # convert % object to float
    df['int_rate'] = df['int_rate'].map(lambda x: float(re.sub(r'%', '', x)) if not pd.isnull(x) else x)
    df['revol_util'] = df['revol_util'].map(lambda x: float(re.sub(r'%', '', x)) if not pd.isnull(x) else x)
    df['zip_code'] = df['zip_code'].map(lambda x: x[0] if not pd.isnull(x) else x)

    stime = dt.datetime(2016,1,1,0,0,0)
    timelist = ['earliest_cr_line', 'last_pymnt_d', 'last_credit_pull_d', 'next_pymnt_d', 'issue_d']
    deslist = ['emp_title', 'desc', 'title']

    for icol in cols:
        index = pd.isnull(df[icol])
        if df[icol].dtype == 'float64':
            tmp = np.mean(df[icol][~index])
            df[icol] = df[icol].map(lambda x: tmp if pd.isnull(x) else x)
        else:
            index = pd.isnull(df[icol])
            counter = Counter(df[icol][~index])
            tmp = counter.most_common()[0][0]
            if not icol in timelist:
                tmp = 'XXX'
            df[icol] = df[icol].map(lambda x: tmp if pd.isnull(x) else x)
            
            if icol in timelist:
                # convert timestamp to number
                df[icol] = df[icol].map(lambda x: -(dt.datetime.strptime(x, '%b-%Y')-stime).days)
            elif icol in deslist:
                df[icol] = df[icol].map(lambda x: 0 if pd.isnull(x) else 1)
            elif icol != 'loan_status':
                cates = df[icol].unique().tolist()
                dicts = {cates[i]: i for i in range(len(cates))}
                df[icol] = df[icol].map(lambda x: dicts[x])
    return
