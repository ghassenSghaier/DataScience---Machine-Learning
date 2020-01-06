# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 14:16:08 2019

@author: LENOVO
"""
#Import the librarys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset =pd.read_csv('Data.csv') 
X = dataset.iloc[:,:3]
Y = dataset.iloc[:,-1]
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(X.iloc[:,1:3])
X.iloc[:,1:3]=imputer.transform(X.iloc[:,1:3])
from sklearn.preprocessing import LabelEncoder , OneHotEncoder 
labelEncoder_X=LabelEncoder()
X.iloc[:,0]=labelEncoder_X.fit_transform(X.iloc[:,0])
onehotencoder=OneHotEncoder(categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()
labelEncoder_Y=LabelEncoder()
Y=labelEncoder_Y.fit_transform(Y)
 from sklearn.model_selection import train_test_split
X_train , X_test , Y_train , Y_test =train_test_split(X,Y,test_size=0.2,random_state=0)
plt.hist(X[:,3])
plt.title('Histogram',fontsize=10)
plt.show()
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)