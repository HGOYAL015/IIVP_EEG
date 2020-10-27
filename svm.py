import re
import nltk
from nltk.stem import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from PIL import _imaging
from sklearn.svm import SVC
from numpy import genfromtxt

import csv
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
import pandas as pd

import numpy as np


def cor_selector(X, y, num_feats):
    cor_list = []
    feature_name = X.columns.tolist()
    # calculate the correlation with y for each feature
    for i in X.columns.tolist():
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append(cor)
    # replace NaN with 0
    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    # feature name
    cor_feature = X.iloc[:, np.argsort(
        np.abs(cor_list))[-num_feats:]].columns.tolist()
    # feature selection? 0 for not select, 1 for select
    cor_support = [True if i in cor_feature else False for i in feature_name]
    return cor_support, cor_feature


data = pd.read_csv('out.csv')
df = data
df.drop(columns=['Label'])
X = df
y = data['Label']
print(np.shape(X))
num_feats = 600
cor_support, cor_feature = cor_selector(X, y, num_feats)
# print(cor_support)
# print(cor_feature)

print(str(len(cor_feature)), 'selected features')
# my_data = genfromtxt('out.csv', delimiter=',')

my_data = pd.read_csv('out.csv')

df = my_data[cor_feature]
df.drop(columns=['Label'])

X_ori = np.array(df)
print(np.shape(X_ori))
Y_ori = np.array(my_data)
print(type(my_data))
# print(my_data)

data_length = 2479

train_len = (int)(data_length*(0.7))
test_len = (data_length-train_len)

# print(my_data)

# y_train=my_data[:,-1]
y_train = Y_ori[1:train_len, -1]

y_test = Y_ori[train_len:, -1]
# print(ans)

# print(ans[0])

X_train = X_ori[1:train_len, :-1]
# X_train=X_ori[:,:-1]

X_test = X_ori[train_len:, :-1]

# print(X_ori)

eeg_svc = SVC(C=1.0, kernel="linear")
# print(type(eeg_svc))

eeg_svc.fit(X_train, y_train.ravel())
print("Training Accuracy:", (eeg_svc.score(X_train, y_train.ravel()))*100, "%")


eeg_svc.predict(X_test)
print("Test Accuracy:", (eeg_svc.score(X_test, y_test.ravel()))*100, "%")
