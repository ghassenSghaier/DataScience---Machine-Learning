# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 16:39:27 2019

@author: LENOVO
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
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X,Y)
plt.scatter(X,Y,color='red')
plt.plot(X,regressor.predict(X),color='blue')
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=12)
X_ploy=poly_reg.fit_transform(X)
regressor.fit(X_ploy,Y)
plt.scatter(X,Y,color='red')
plt.plot(X,regressor.predict(X_ploy),color='blue')
a=regressor.intercept_
b=regressor.coef_
print (a,b)
MSE= metrics.mean_squared_error (regressor.predict(X_ploy),Y)
RMSE =MSE**0.5
print (MSE,RMSE)
#Feature Scaling 
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
sc_Y=StandardScaler()
X=sc_X.fit_transform(X)

Y=sc_Y.fit_transform(Y.values.reshape(-1,1))

#Visualizing the SVR
from sklearn.svm import SVR 
regressor1= SVR(kernel='rbf')
regressor1.fit(X,Y)
Z_pred= regressor1.predict(X)
plt.scatter(X,Y,color='red')
plt.plot(X,Z_pred,color='blue')
S=np.array([6.5]).reshape(1,-1)
Y_pred=regressor1.predict(sc_X.transform(S))
H=sc_Y.inverse_transform(regressor1.predict(sc_X.transform(S)))
H2=sc_Y.inverse_transform(regressor1.predict(X))
MSE= metrics.mean_squared_error (H2,Y)
RMSE =MSE**0.5
print (MSE,RMSE)
#Fitting Decision Tree regression to the dataset
from sklearn.tree import DecisionTreeRegressor 
regressor2=DecisionTreeRegressor(criterion="mse")
regressor2.fit(X,Y)
X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,Y,color='red')
plt.plot(X_grid,regressor2.predict(X_grid),color='blue')