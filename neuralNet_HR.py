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
HRdata = pd.read_csv("HRdata.csv")
X = HRdata.iloc[:, 0:9]
y = HRdata.iloc[:, 9]

# encode categorical variables
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
X_labelencoder = LabelEncoder()
X[:, [7:9]] = X_labelencoder.fit_transform(X[:, [7:9]])
encoder = OneHotEncoder(categorical_features = [7,9])
X = encoder.fit_transform(X).toarray()

# split data into training and testing datasets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.25, 
                                                    random_state=0)


# feature scaling (required for ANN)
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X_train)

### BUILD THE ARTIFICIAL NEURAL NETWORK ################

### EVALUATE ACCURACY WITH TEST SET ####################


### TEST THE NEURAL NET WITH EXAMPLE DATA ##############
