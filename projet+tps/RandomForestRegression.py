# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 23:02:26 2020

@author: lenovo
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

dataset=pd.read_csv("Position_Salaries.csv",sep=',')
X = dataset.iloc[:,1:2]
Y = dataset.iloc[:,-1]
plt.scatter(X,Y,color="red")
plt.title("hauteur en fonction de diametre")
plt.xlabel('diametre')
plt.ylabel('hauteur')
plt.show()

#fitting random forest regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=100,random_state=0)
regressor.fit(X,Y)
X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,Y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')

X_grid=np.arange(min(X),max(X),0.01)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,Y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')

Z_pred= regressor.predict(X)
plt.scatter(X,Y,color='red')
plt.plot(X,Z_pred,color='blue')
S=np.array([6.5]).reshape(1,-1)
Y_pred=regressor.predict(S)