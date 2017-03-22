import numpy as np
import pandas as pd
from sklearn.datasets import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import cross_val_score
from sklearn.base import clone
from numpy import float32

rng = np.random.RandomState(0)
 
 
#Load the data into DataFrames
train_users = pd.read_csv('C:/Users/Home/Desktop/train.csv')
test_users = pd.read_csv('C:/Users/Home/Desktop/test.csv')

df = pd.DataFrame (train_users)
del df['v125']


dataset = df
dataset.initarget = df['target']

dataset.data = dataset


X_fulln = dataset.data
del X_fulln['target']
del X_fulln['ID']
del X_fulln['v3']
del X_fulln['v22']
del X_fulln['v24']
del X_fulln['v30']
del X_fulln['v31']
del X_fulln['v56']
del X_fulln['v113']
del X_fulln['v112']
del X_fulln['v107']
del X_fulln['v110']
del X_fulln['v91']
del X_fulln['v47']
del X_fulln['v52']
del X_fulln['v66']
del X_fulln['v71']
del X_fulln['v74']
del X_fulln['v75']
del X_fulln['v79']


y_full = dataset.initarget[:-1000]

X_full = X_fulln[:-1000]

n_samples = X_full.shape[0]
n_features = X_full.shape[1]
 
n_trees = 100
 
#print("The shape of the dataset is" %s % str(X_full.shape))
#print("The number of trees for this benchmarking is" %s % n_trees)

#X_full = X_full.astype(float64)
# Estimate the score on the entire dataset, with no missing values
estimator = RandomForestClassifier(random_state=0, n_estimators=n_trees, n_jobs=2)
score = cross_val_score(estimator, X_full, y_full).mean()
print ("Score with the entire dataset = %.2f" % score)
 
X_missing = X_full.copy()
y_missing = y_full.copy()
score_impute_missing = []
score_ignore_missing = []
score_internally_handled = []
true_miss_percent = []
 
n_values = n_samples * n_features
 
# Type of missing value
MCAR = True
NMAR = False
MAR = False
 
unique_classes = np.unique(y_full)
 
# Pick a feature at random
# 30% of features have missing values
feature_picked = [np.random.randint(0, n_features) for i in range(n_features/3)]
# Pick a class at random
class_picked = [unique_classes[np.random.randint(0, len(unique_classes))] for i in range(n_features/3)]
 
#print feature_picked, class_picked
 
n_estimators = 100
 
 
# Try to miss 3% of data at each iteration
miss_rate = 0.003
# (or n_missing_vals values)
n_missing_vals = int(n_pts * miss_rate)
 
for i in range(20):
   
    if MCAR:
        missing_samples = np.hstack([np.ones(n_missing_vals), np.zeros(n_pts - n_missing_vals)])
        print (missing_samples.size, X_full.size, X_full.shape)
        np.random.shuffle(missing_samples)
        missing_samples = missing_samples.reshape(X_full.shape).astype(bool)
        X_missing[missing_samples] = np.nan
       
    elif NMAR:
        for f, c in zip(feature_picked, class_picked):
            missing_samples_f = np.where(y_full == c)[0]
            # Shuffle the indices
            np.random.shuffle(missing_samples_f)
            print (missing_samples_f)
           
            n_miss_per_feature = n_missing_vals / (n_features/3)  # spread across 30% of features
            pick = np.random.choice(len(missing_samples_f),
                                    n_miss_per_feature
                                    if len(missing_samples_f) > n_miss_per_feature
                                    else len(missing_samples_f)/2)
            missing_samples_f = missing_samples_f[pick]
           
            print (X_missing[(missing_samples_f[:n_missing_vals]), feature_picked])
            X_missing[missing_samples_f, f] = np.nan
            print (X_missing[(missing_samples_f[:n_missing_vals]), feature_picked])
            #break
 
            print (X_missing, y_missing)
            #break
       
    elif MAR:
        pass
   
    # missing_samples might overlap
    n_true_missing = np.sum(np.isnan(X_missing))
    mp.append(n_true_missing*100.0/n_pts*1.0)
    # The percentage of missing values
    #mp.append(i)
    print ("%f percent of data Missing" % mp[-1])
   
    # Estimate the score without the samples containing missing values
    X_filtered = X_full[np.where(np.logical_not(np.isnan(X_missing)))[0], :]
    y_filtered = y_full[np.where(np.logical_not(np.isnan(X_missing)))[0]]
    estimator = RandomForestClassifier(random_state=0, n_estimators=n_estimators, missing_values='NaN')
    score = cross_val_score(estimator, X_filtered, y_filtered).mean()
    score_miss.append(score)
    print("Score without the samples containing missing values = %.2f" % score)
   
    # Imputing the missing values
    estimator = Pipeline([("imputer", Imputer(missing_values='NaN',
                                              strategy="mean",
                                              axis=0)),
                          ("forest", RandomForestClassifier(
                                         random_state=0,
                                         n_estimators=n_estimators))])
 
    score = cross_val_score(estimator, X_missing, y_full).mean()
    score_imp.append(score)
    print("Score after imputation of the missing values = %.2f" % score)
   
    # Internally handle missing values
    estimator = RandomForestClassifier(random_state=0, n_estimators=n_estimators, missing_values='NaN')
    score = cross_val_score(estimator, X_missing, y_full).mean()
    score_cv.append(score)
    print("Score when missing values are internally handled by DTC = %.2f" % score)
    #print
   
   
from matplotlib import pyplot as plt
 
fig, ax = plt.subplots()
fig.set_size_inches(10, 10)
 
 
ax.plot(mp, np.array(score_miss), c = "r", marker = "o", linestyle="--", label = "Ignoring samples with MV")
ax.plot(mp, np.array(score_imp), c = "b", marker = "o", linestyle="--", label = "Imputing MV")
ax.plot(mp, np.array(score_cv), c = "g", marker = "o", linestyle="--", label = "Internally handle MV")
 
ax.set_xlabel("% of Missing data")
ax.set_ylabel("cross-val score")
 
ax.legend(loc='best')
 
plt.show()
