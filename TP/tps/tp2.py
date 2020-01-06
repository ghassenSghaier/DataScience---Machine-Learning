# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 15:35:55 2019

@author: LENOVO
"""

import pandas as pd 
dataset=pd.read_csv('Salary_Data.csv',sep=';')
X=dataset.iloc[:,:-1]
Y=dataset.iloc[:,-1]
import matplotlib.pyplot as plt
plt.scatter(X,Y,color='red')
plt.title('Salary vs Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
import numpy as np
from sklearn.model_selection import train_test_split
X_train , X_test , Y_train , Y_test =train_test_split(X,Y,test_size=1.0/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred = regressor.predict(X_train)
plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train,Y_pred,color='blue')
plt.title('Salary vs Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
from sklearn import metrics
MAE= metrics.mean_absolute_error(Y_pred,Y_train)
MSE= metrics.mean_squared_error(Y_pred,Y_train)
RMSE= metrics.mean_absolute_error(Y_pred,Y_train)**0.5
print(MAE)
print(MSE)
print(RMSE)
x=dataset.iloc[:,0].values
y=dataset.iloc[:,1].values
r=(np.mean(x*y)-np.mean(x)*np.mean(y))/(np.std(x)*np.std(y))
sigma=np.corrcoef(x,y)


regressor = LinearRegression()
regressor.fit(X_test,Y_test)
Y_pred = regressor.predict(X_test)
plt.scatter(X_test,Y_test,color='red')
plt.plot(X_test,Y_pred,color='blue')
plt.title('Salary vs Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

from sklearn import metrics
MAE= metrics.mean_absolute_error(Y_pred,Y_test)
MSE= metrics.mean_squared_error(Y_pred,Y_test)
RMSE= metrics.mean_absolute_error(Y_pred,Y_test)**0.5
print(MAE)
print(MSE)
print(RMSE)
