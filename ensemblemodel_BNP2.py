import numpy as np
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from brew.base import Ensemble
from brew.base import EnsembleClassifier
from brew.combination.combiner import Combiner
#=====================================================================
train = pd.read_csv('C:/Users/Home/Desktop/train.csv')

target = train['target'].values

df = pd.DataFrame(train)
del df['ID']
del df['target']

test = pd.read_csv('C:/Users/Home/Desktop/test.csv')

id_test = test['ID'].values

df1 = pd.DataFrame(test)

del df1['ID']

for (train_name, train_series), (test_name, test_series) in zip(train.iteritems(),test.iteritems()):
    if train_series.dtype == 'O':
        #for objects: factorize
        train[train_name], tmp_indexer = pd.factorize(train[train_name])
        test[test_name] = tmp_indexer.get_indexer(test[test_name])
        #but now we have -1 values (NaN)
    else:
        #for int or float: fill NaN
        tmp_len = len(train[train_series.isnull()])
        if tmp_len>0:
            #print "mean", train_series.mean()
            train.loc[train_series.isnull(), train_name] = -999 
        #and Test
        tmp_len = len(test[test_series.isnull()])
        if tmp_len>0:
            test.loc[test_series.isnull(), test_name] = -999

train = train
test = test

#=========================================================================================================
#Load transformed numeric and categoric data
#Data transformed following correlation analysis, feature selection
#And where necessary PCA to reduce the dimension of datasets

train_num_new = pd.read_csv('C:/Users/Home/Desktop/train_num_new.csv')
test_num_new = pd.read_csv('C:/Users/Home/Desktop/test_num_new.csv')
train_cat_new = pd.read_csv('C:/Users/Home/Desktop/train_cat_new.csv')
test_cat_new = pd.read_csv('C:/Users/Home/Desktop/test_cat_new.csv')

#=========================================================================================================

train_cat_new = pd.DataFrame(train_cat_new) #to be concatenated with train_num_new
test_cat_new = pd.DataFrame(test_cat_new) #to be concatenated with test_num_new
train_num_new = pd.DataFrame(train_num_new)
test_num_new = pd.DataFrame(test_num_new)

train_new = pd.concat([train_num_new,train_cat_new],axis=1) #new train data for classification model
test_new = pd.concat([test_num_new,test_cat_new],axis=1) #new train data for prediction
target = target

#Correlation analysis shows that v50,v129,v10,v14 have strong correlation to the target
#polynomial expansion of these features and then added to the rest to improve
#prediction accuracy

train = pd.DataFrame(train)
test = pd.DataFrame(test)

from sklearn.preprocessing import StandardScaler

target_corr_train = train.ix[:,['v50','v129','v10','v14']]
#target_corr_train = StandardScaler().fit_transform(target_corr_train)
target_corr_test = test.ix[:,['v50','v129','v10','v14']]
#target_corr_test = StandardScaler().fit_transform(target_corr_test)

v502 = target_corr_train['v50']**2
v1292 = target_corr_train['v129']*2
v102 = target_corr_train['v10']*2
v142 = target_corr_train['v14']*2
df_train = np.column_stack((v502,v1292,v102,v142))
df_train1 = pd.DataFrame(df_train)
df_train1.columns = ['v50_new', 'v129_new','v10_new','v14_new']

#poly = PolynomialFeatures(2)

train_new2 = pd.concat([train_new, df_train1],axis=1)

#train_poly = poly.fit_transform(train_new2)

v501 = target_corr_test['v50']*2
v1291 = target_corr_test['v129']*2
v101 = target_corr_test['v10']*2
v141 = target_corr_test['v14']*2
df_test = np.column_stack((v501,v1291,v101,v141))
df_test1 = pd.DataFrame(df_test)
df_test1.columns = ['v50_new', 'v129_new','v10_new','v14_new']

test_new2 = pd.concat([test_new, df_test1],axis=1)

#test_poly = poly.fit_transform(test_new2)

#===============================================================================
print("Building multiple models...")

X_train, X_test, y_train, y_test = train_test_split(train_new2, target, test_size=0.5)


clf4 = ExtraTreesClassifier(bootstrap=False, class_weight='auto', criterion='entropy',
           max_depth=None, max_features='sqrt', max_leaf_nodes=None,
           min_samples_leaf=1, min_samples_split=4,
           min_weight_fraction_leaf=0.0, n_estimators=250, n_jobs=1)

clf4 = clf4.fit(X_train, y_train)
y_pr4 = clf4.predict_proba(X_test)
print("Extra tree logloss=",log_loss(y_test, y_pr4[:,1]))

clf3 = GradientBoostingClassifier(init=None, learning_rate=0.1, loss='deviance',
              max_depth=2, max_features=None, max_leaf_nodes=None,
              min_samples_leaf=1, min_samples_split=3,
              min_weight_fraction_leaf=0.0, n_estimators=300)

clf3 = clf3.fit(X_train, y_train)
y_pr3 = clf3.predict_proba(X_test)
print("Gradient boost logloss=",log_loss(y_test, y_pr3[:,1]))


exit()

#=========================================================================================================================

y_pred_test = est.predict_proba(test_poly)

print("Test data predictions...",y_pred_test[:,1])

print('Saving the predictions to csv file...you have done well to wait')

pd.DataFrame({"ID": id_test, "PredictedProb": y_pred_test[:,1]}).to_csv('new.csv',index=False)


print("Prediction of test data completed")

#===============================================================================================
