# importing the libraries
from torch.optim import Adam, SGD
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.autograd import Variable
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# for reading and displaying images
from skimage.io import imread
import matplotlib.pyplot as plt


# for creating validation set

# for evaluating the model

# PyTorch libraries and modules


data = pd.read_csv("out.csv")
df = data.drop(columns='Label')

X_ori = np.array(df)
Y_ori = np.array(data['Label'])

data_length = 2479

train_len = (int)(data_length*(0.7))
test_len = (data_length-train_len)

trainX = X_ori[:train_len, ].reshape(
    X_ori.shape[0], 1, 28, 28).astype('float32')
X_train = trainX / 255.0

y_train = Y_ori[:, ]

print(np.shape(X_train))
print(y_train)


# # Reshape and normalize test data
# testX = test[:, 1:].reshape(test.shape[0], 1, 28, 28).astype('float32')
# X_test = testX / 255.0

# y_test = test[:, 0]
