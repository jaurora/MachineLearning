import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib as mpl
mpl.rcParams['interactive'] = True
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

def param(sc_train, sc_test, depths, etas, rounds):
### plot score vs parameters of the model ##### 
### sc_train: training score
### sc_test: test score

    # Four subplots, the axes array is 2-d
    f, axarr = plt.subplots(2, 2)

    X, Y = np.meshgrid(depths, rounds)
    levels = np.linspace(np.min(sc_test), np.max(sc_test), 30)
    cs = axarr[0, 0].plot(X, Y, sc_test, levels, cmap=cm.get_cmap('coolwarm'))
    cb = fig.colorbar(cs, ax=axs[0], format="%.2f")
    cb.set_label('meters')


    plt.show()

    plt.savefig('feature_importance.png')


def feature_importance(X, y, xtab):
### Plot Feature Importance ##### 
### X: matrix of features
### y: classification
### xtab: feature list

    # Buid a random forest
    clf = RandomForestClassifier(n_estimators=100)
    clf = clf.fit(X, y)

    # Plot the feature importances of the tree forest
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    ind = np.arange(len(importances))  # the x locations for the groups
    plt.figure(figsize=(14,8))
    plt.bar(ind[:10], importances[indices][:10], color="r", align="center")
    plt.ylabel('Feature Importance')
    plt.xlabel('Feature')
    plt.title('Top 10 Important Features')
    plt.xticks(ind, xtab[indices][:10])
    plt.xlim([-1, 10])
    plt.show()
    plt.savefig('feature_importance.png')


def fp_ratio(cdf):
### Plot Fully Paid ratio Predicted by model ##### 
### cdf: dataframe with loan_status (predicted), cur_loan_status (current)

    # separate the dataset by current loan status
    sts = cdf['cur_loan_status'].unique()
    bars = np.zeros((len(sts)))
    for idx in range(len(sts)):
        subselect = cdf['cur_loan_status'] == sts[idx]
        subset = cdf['loan_status'][subselect]
        bars[idx] = len(subset[subset == 'Fully Paid']) / float(len(subset))
        
    # plot Fully Paid ratio

    ind = np.arange(len(bars))  # the x locations for the groups
    plt.figure(figsize=(10,6))
    plt.title("Fully Paid ratio")
    plt.bar(ind, bars, color="r", align="center")
    plt.ylabel('Fully Paid Ratio')
    plt.xlabel('Current Loan Status')
    plt.xticks(ind, sts.tolist())
    plt.xlim([-1, 5])
    plt.show()
    plt.savefig('fullypaid_ratio.png')
