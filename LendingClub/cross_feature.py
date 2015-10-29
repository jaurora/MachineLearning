import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import math
from collections import Counter
import matplotlib as mpl
mpl.rcParams['interactive'] = True
import datetime as dt

def cross_hist(c1, c2):
    if  (c1.describe().name in ['url']):
        print('Histogram Not Available!')
    
    elif  (c1.describe().name in ['emp_title', 'desc', 'title']):
        cn = [0 for ifea in range(2)]
        fn = [0 for ifea in range(2)]

        index = (c2 == 'Fully Paid') & (pd.isnull(c1))
        fn[0] = len(c1[index])/float(len(c1))
        index = (c2 == 'Fully Paid') & (~pd.isnull(c1))
        fn[1] = len(c1[index])/float(len(c1))

        index = (c2 == 'Charged Off') & (pd.isnull(c1))
        cn[0] = len(c1[index])/float(len(c1))
        index = (c2 == 'Charged Off') & (~pd.isnull(c1))
        cn[1] = len(c1[index])/float(len(c1))

        ind = np.arange(2)    # the x locations for the groups
        width = 0.35                   # the width of the bars: can also be len(x) sequence

        fn = np.array(fn)
        cn = np.array(cn)
        div = fn + cn
        fn /= div
        cn /= div

        p1 = plt.bar(ind, fn,   width, color='y')
        p2 = plt.bar(ind, cn, width, color='r', bottom=fn)

        plt.ylabel('Percentages')
        plt.title('Percentages by Availability of '+ c1.describe().name)
        plt.xticks(ind+width/2., ['W/O', 'W'])
        plt.legend( (p1[0], p2[0]), ['Fully Paid', 'Charged Off'])
        plt.grid(True)
        plt.show() 
            

    elif (c1.dtype == 'float64' or c1.describe().name 
          in ['int_rate', 'revol_util', 'annual_inc', 'earliest_cr_line', 
              'last_pymnt_d', 'last_credit_pull_d', 'next_pymnt_d']):
        mask = pd.isnull(c1) | pd.isnull(c2)
        c1 = c1[~mask]  
        c2 = c2[~mask]
        if (c1.describe().name in ['int_rate', 'revol_util']):
            c1 = c1.map(lambda x: float(re.sub(r'%', '', x)) if not pd.isnull(x) else x)
        elif (c1.describe().name == 'annual_inc'):
            c1 = c1.map(lambda x: math.log10(x))
        elif (c1.describe().name in ['earliest_cr_line', 'last_pymnt_d', 'last_credit_pull_d', 'next_pymnt_d']):
            # Set a month to obtain the credit history
            stime = dt.datetime(2016,1,1,0,0,0)
            c1 = c1.map(lambda x: -(dt.datetime.strptime(x, '%b-%Y')-stime).days)

        fval = c1[c2 == 'Fully Paid']
        cval = c1[c2 == 'Charged Off']
           
        fweights = np.ones_like(fval)/float(len(c1))
        cweights = np.ones_like(cval)/float(len(c1))            
        fn, bins = np.histogram(fval.values, 10, weights=fweights)            
        cn, bins = np.histogram(cval.values, bins, weights=cweights)

        ind = np.arange(len(fn))    # the x locations for the groups
        width = 0.35                   # the width of the bars: can also be len(x) sequence

        fn = np.array(fn)
        cn = np.array(cn)
        div = fn + cn
        index = np.where(div > 1.e-10)
        fn[index] /= div[index] 
        cn[index]  /= div[index] 
            
        p1 = plt.bar(ind, fn,   width, color='y')
        p2 = plt.bar(ind, cn, width, color='r', bottom=fn)

        plt.ylabel('Percentages')
        if (c1.describe().name == 'annual_inc'):
            plt.title('Percentages by Log10 '+ c1.describe().name)
        elif (c1.describe().name in 
              ['earliest_cr_line', 'last_pymnt_d', 'last_credit_pull_d']):
            plt.title('Percentage by Days Since [ '+c1.describe().name + ' ] to 1/1/2016')
        else:
            plt.title('Percentages by '+ c1.describe().name)

        strings = ["%.2f" % number for number in bins]
        plt.xticks(ind+width/2., strings)
        plt.legend( (p1[0], p2[0]), ['Fully Paid', 'Charged Off'])
        plt.grid(True)
        plt.show() 

    elif c1.dtype == 'O':
        if (c1.describe().name == 'zip_code'):
            c1 = c1.map(lambda x: re.sub(r'\D', '', x)[0])
        catos = c1.unique().tolist()
        cn = [0 for ifea in range(len(catos))]
        fn = [0 for ifea in range(len(catos))]
        decs = c1.describe().name
        for icato in range(len(catos)):
            index = (c2 == 'Fully Paid') &  (c1 == catos[icato])
            fn[icato] = len(c1[index])/float(len(c1))
            index = (c2 == 'Charged Off') & (c1 == catos[icato])
            cn[icato] = len(c1[index])/float(len(c1))

        ind = np.arange(len(catos))    # the x locations for the groups
        width = 0.35                   # the width of the bars: can also be len(x) sequence

        fn = np.array(fn)
        cn = np.array(cn)
        div = fn + cn
        fn /= div
        cn /= div

        p1 = plt.bar(ind, fn,   width, color='y')
        p2 = plt.bar(ind, cn, width, color='r', bottom=fn)

        plt.ylabel('Percentages')
        if (c1.describe().name == 'zip_code'):
            plt.title('Percentages by head of '+ c1.describe().name)
        else:
            plt.title('Percentages by '+ c1.describe().name)
        plt.xticks(ind+width/2., catos )
        plt.legend( (p1[0], p2[0]), ['Fully Paid', 'Charged Off'])
        plt.grid(True)
        plt.show() 

    svn = c1.describe().name + '_by_status'
    svp = input('Save figure in ' + svn + '.png? [ Yes->1 / No->0 ]: ')
    if svp == 1:
        plt.savefig(svn + '.png')

    raw_input("Press Enter to continue...")           
    plt.close()

    return

