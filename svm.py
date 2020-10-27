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

my_data = genfromtxt('out.csv', delimiter=',')
print(type(my_data))

data_length=2479

train_len=(int)(data_length*(0.7))
test_len=(data_length-train_len)

# print(my_data)

# y_train=my_data[:,-1]
y_train=my_data[1:train_len,-1]

y_test=my_data[train_len:,-1]
# print(ans)

# print(ans[0])

X_train=my_data[1:train_len,:-1]
# X_train=my_data[:,:-1]

X_test=my_data[train_len:,:-1]

# print(my_data)

spam_svc = SVC(C=1.0,kernel ="linear")
# print(type(spam_svc))

spam_svc.fit(X_train,y_train.ravel())
print("Training Accuracy:",(spam_svc.score(X_train,y_train.ravel()))*100,"%")


spam_svc.predict(X_test)
print("Test Accuracy:",(spam_svc.score(X_test,y_test.ravel()))*100,"%")







