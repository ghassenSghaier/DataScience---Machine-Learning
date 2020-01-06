# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 23:04:42 2020

@author: lenovo
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

dataset=pd.read_csv("Social_Network_Ads.csv")
X = dataset.iloc[:,[2,3]].values
Y = dataset.iloc[:,-1]

from sklearn.model_selection import train_test_split
X_train , X_test , Y_train , Y_test =train_test_split(X,Y,test_size=0.25,random_state=0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train=sc.fit_transform(X_train)

X_test=sc.transform(X_test)
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,Y_train)

Y_pred =classifier.predict(X_test)
cm=confusion_matrix(Y_test,Y_pred)