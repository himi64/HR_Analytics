# -*- coding: utf-8 -*-
"""
# filename: ann_HR
# author: Himanshu Makharia
# description: artificial neural network to analyze 

install keras, tensorflow, theano
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### DATA PREPROCESSING #################################

# import dataset
data = pd.read_csv("HRdata.csv")
HRdata = np.array(data) # convert to numpy array for slicing

X = HRdata[:, 0:9]
y = HRdata[:, 9]

# encode categorical variables
HRdata.dtypes
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 7] = labelencoder_X.fit_transform(X[:, 7])
X[:, 8] = labelencoder_X.fit_transform(X[:, 8])

# one hot encoding: input for this must be a matrix of integers
onehot1 = OneHotEncoder(categorical_features = [7])
onehot2 = OneHotEncoder(categorical_features = [8])
X = onehot1.fit_transform(X).toarray() # encode department column
X = onehot2.fit_transform(X).toarray()


# split data into training and testing datasets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.25, 
                                                    random_state=0)

# feature scaling (required for ANN)
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)

### BUILD THE ARTIFICIAL NEURAL NETWORK ################

import keras
from keras.models import Sequential
from keras.layers import Dense

# initialize the ANN
classifier = Sequential()

# add first layer (inputs)
classifier.add(Dense(units = 10, 
                     kernel_initializer = 'uniform', 
                     activation = 'relu', 
                     input_dim = 19)) # 9 nodes in the hidden layer, 19 inputs 
                     #NB: (input nodes + output nodes) / 2 = units (# of nodes)

# add second layer
classifier.add(Dense(units = 10, 
                     kernel_initializer = 'uniform',
                     activation = 'relu'))

# add output layer
classifier.add(Dense(units=1,
                     kernel_initializer = 'uniform',
                     activation = 'sigmoid'))

# compiling the ANN
classifier.compile(optimizer = 'adam', 
                   loss = 'binary_crossentropy',
                   method = ['accuracy'])

# fit the ANN on the training & testing set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 50)


### EVALUATE ACCURACY WITH TEST SET ####################


### TEST THE NEURAL NET WITH EXAMPLE DATA ##############
