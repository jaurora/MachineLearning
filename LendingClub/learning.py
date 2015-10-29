import numpy as np
import xgboost as xgb
from sklearn import ensemble, preprocessing, cross_validation
from sklearn.metrics import roc_auc_score as auc

### training model ##### 
def learning(X, y, depth, eta, rounds, subs=1.0):
 
    rng = np.random.RandomState()
    skf = cross_validation.StratifiedKFold(y, n_folds=3, shuffle=True, random_state=rng)
    trscores, cvscores = [], []

    num_round = rounds
    param = {'max_depth':depth, 'eta':eta, 'sub_sample': subs, 'silent':1, 'objective':'binary:logistic' }

    for train_index, test_index in skf:
        print("TRAIN:", train_index, "TEST:", test_index)

        #### cross validations #####  
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        watchlist  = [(dtest,'eval'), (dtrain,'train')]

        bst = xgb.train(param, dtrain, num_round, watchlist)
        ptrain = bst.predict(dtrain)
        ptest  = bst.predict(dtest)

        trscore = auc(y_train, ptrain)
        cvscore = auc(y_test,  ptest)
        trscores.append(trscore)
        cvscores.append(cvscore)
        
    return np.mean(trscores), np.mean(cvscores), bst
