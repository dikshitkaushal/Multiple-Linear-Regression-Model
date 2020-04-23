# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 22:01:46 2020

@author: DELL
"""

"Importing the libraries"
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"importing the dataset"
dataset = pd.read_csv("50_Startups.csv")
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

"encode the categorical data "
"ONE HOT ENCODING"
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('Encoder',OneHotEncoder(),[3])],remainder="passthrough")
x=np.array(ct.fit_transform(x))

"splitting the dataset into training and test set"
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0) 

"training the model on the training set"
from sklearn.linear_model import LinearRegression
regressor =LinearRegression()
regressor.fit(x_train,y_train)

"predicting the test set results"
y_pred=regressor.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1)) 
"Careful of braces"
