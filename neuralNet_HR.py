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

# X.dtypes

X = HRdata[:, 0:9]
y = HRdata[:, 9]

# encode categorical variables
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
X_test = sc.fit_transform(X_test)

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
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5) # threshold for left or no left

# create confusion matrix (needs to be fixed)
from sklearn.metrics import confusion_matrix
prediction = []
for row in y_pred:
    if row > 0.5:
        prediction.append([1])
    else: prediction.append([0])  
    

y_test = pd.to_numeric(y_test)

cm = confusion_matrix(prediction, y_test)
# accuracy = (2812 + 772)/(2812 + 772+69+97) = 95.57%

### TEST THE NEURAL NET WITH EXAMPLE DATA ##############

'''
dept: col 0 & 9 (sales)
satisfaction_level: 0.2
last_evaluation: 0.3
number_project: 2
average_montly_hours: 200
time_spend_company: 1
Work_accident: 0
promotion_last_5years: 0
salary: 1 (low)
'''

sample_row = sc.transform(np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0.2, 0.3, 2, 200, 1, 0, 0, 1]]))
sample_predict = classifier.predict(sample_row)