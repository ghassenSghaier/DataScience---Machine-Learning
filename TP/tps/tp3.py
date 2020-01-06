# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 14:49:35 2019

@author: LENOVO
"""

import pandas as pd 
dataset=pd.read_excel('Europe.xlsx')
Y=dataset.iloc[:,2].values
X=dataset.iloc[:,3].values
from sklearn.model_selection import train_test_split
X_train , X_test , Y_train , Y_test =train_test_split(X,Y,test_size=1.0/3,random_state=0)
import matplotlib.pyplot as plt
plt.scatter(X,Y,color='red')
plt.title('population vs superficie')
plt.xlabel('superficie')
plt.ylabel('population')
plt.show()
import numpy as np
r = (np.mean(X*Y)- np.mean(X)* np.mean(Y))/(np.std(X)*np.std(Y))


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
x_train_1 = X_train.reshape(-1,1)
regressor.fit(x_train_1,Y_train)
y_pred=regressor.predict(x_train_1)
plt.scatter(x_train_1,Y_train,color='red')
plt.plot(x_train_1,y_pred,color='blue')
from sklearn import metrics 
MAE =metrics.mean_absolute_error(y_pred,Y_train)
MSE= metrics.mean_squared_error (y_pred,Y_train)
RMSE =MSE**0.5
print (MAE,MSE)
print (RMSE)
a=regressor.intercept_
b=regressor.coef_
print (a,b)

regressor = LinearRegression()
x_test_1 = X_test.reshape(-1,1)
regressor.fit(x_test_1,Y_test)
y_pred_test=regressor.predict(x_test_1)
plt.scatter(x_test_1,Y_test,color='red')
plt.plot(x_test_1,y_pred_test,color='blue')
from sklearn import metrics 
MAE =metrics.mean_absolute_error(y_pred_test,Y_test)
MSE= metrics.mean_squared_error (y_pred_test,Y_test)
RMSE =MSE**0.5
print (MAE,MSE)
print (RMSE)
plt.hist(Y)
x=X.reshape(-1,1)
np.mean(Y/x)
regressor.intercept_
regressor.coef_
