import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import math
from collections import Counter
import matplotlib as mpl
mpl.rcParams['interactive'] = True
import datetime as dt

def vis_feature(val):
    if  (val.describe().name in ['url']):
        print('Histogram Not Available!')
        
    elif  (val.describe().name in ['emp_title', 'desc', 'title']):
        mask = pd.isnull(val)
        cval = val[~mask]
        nval = val[mask]
        arrs = np.array([len(cval), len(nval)])/float(len(val))
        ind = np.arange(2)
        width = 0.4
        fig, ax = plt.subplots()
        rects = ax.bar(ind, arrs, width, color='g')
        ax.set_xlabel(val.describe().name + ' Availability')
        ax.set_ylabel('Ratio')
        ax.set_xticks(ind+width/2.)
        ax.set_xticklabels(['Yes', 'No'])
        ax.set_title('Ratio by '+ val.describe().name + ' Availability')
        plt.grid(True)
        plt.show() 
            
        # Statistics
        print('Statistics:')
        print(val.describe().name + ' Availability')
        print('Yes: '+ '%0.3f' % arrs[0])
        print('No : '+ '%0.3f' % arrs[1])
        

    elif (val.dtype == 'float64' or val.describe().name 
          in ['int_rate', 'revol_util','annual_inc', 'earliest_cr_line', 
              'last_pymnt_d', 'last_credit_pull_d', 'next_pymnt_d']):

        mask = pd.isnull(val)
        cval = val[~mask] 

        if (cval.describe().name in ['int_rate', 'revol_util']):
            cval = cval.map(lambda x: float(re.sub(r'%', '', x)) if not pd.isnull(x) else x)
        elif (cval.describe().name == 'annual_inc'):
            cval = cval.map(lambda x: math.log10(x))
        elif (cval.describe().name in ['earliest_cr_line', 'last_pymnt_d', 'last_credit_pull_d', 'next_pymnt_d']):
            # Set a month to obtain the credit history
            stime = dt.datetime(2016,1,1,0,0,0)
            cval = cval.map(lambda x: -(dt.datetime.strptime(x, '%b-%Y')-stime).days)           

        weights = np.ones_like(cval)/float(len(cval))
        plt.figure()
        n, bins, patches = plt.hist(cval.values, bins=20, weights=weights, facecolor='green')            
        plt.xlabel(val.describe().name)
        plt.grid(True)
        plt.ylabel('Ratio')
        if (cval.describe().name == 'annual_inc'):
            plt.title('Percentages by Log10 '+ cval.describe().name)
        elif (cval.describe().name in 
              ['earliest_cr_line', 'last_pymnt_d', 'last_credit_pull_d']):
            plt.title('Percentage by Days Since [ '+ cval.describe().name + ' ] to 1/1/2016')
        else:
            plt.title('Percentages by '+ cval.describe().name)            
        plt.show()

        # Statistics
        print('Statistics:')
        print(val.describe().name)
        tmp = np.array(cval.values)
        print('Mean: '+ '%0.3f' % tmp.mean())
        print('Std : '+ '%0.3f' % np.std(tmp))
        print('NAN Count: '+str(len(val[mask]))+' ['
              +str(round(float(len(val[mask]))/len(val)*100,3))+'%]')
        
    elif val.dtype == 'O':
        if (val.describe().name == 'zip_code'):
            val = val.map(lambda x: re.sub(r'\D', '', x)[0])

        mask = pd.isnull(val)
        cnts = sorted(Counter(val).items())
        feas = []
        fcns = []
        for tup in cnts:
            feas.append(tup[0])
            fcns.append(tup[1])
            
        ind = np.arange(len(feas))
        width = 0.4

        fig, ax = plt.subplots()
        rects = ax.bar(ind, np.array(fcns, dtype='float64')/len(val), width, color='g')

        # add some text for labels, title and axes ticks
        ax.set_ylabel('Ratio')        
        ax.set_xlabel(val.describe().name)
        ax.set_title('Ratio by '+ val.describe().name)

        if (val.describe().name == 'zip_code'):
            ax.set_title('Ratio by head of '+ val.describe().name)
        else:
            ax.set_title('Ratio by '+ val.describe().name)

        ax.set_xticks(ind+width/2.)
        ax.set_xticklabels(feas)
        plt.grid(True)
        plt.show()
        
        # Statistics
        print('Statistics:')
        print(val.describe().name)
        tmp = np.array(fcns, dtype='float64')/len(val)
        for ifea in range(len(feas)):
            print(str(feas[ifea]) +':  '+ '%0.3f' % tmp[ifea])
        print('NAN Count: '+str(len(val[mask]))+' ['
              +str(round(float(len(val[mask]))/len(val)*100,3))+'%]')

    
    svn = val.describe().name
    svp = input('Save figure in ' + svn + '.png? [ Yes->1 / No->0 ]: ')
    if svp == 1:
        plt.savefig(svn + '.png')

    raw_input("Press Enter to continue...")           
    plt.close()

    return

