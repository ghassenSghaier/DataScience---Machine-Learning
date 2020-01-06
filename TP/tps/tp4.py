# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 16:02:25 2019

@author: LENOVO
"""
#importing the libraires
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics 
#imprting the dataset
dataset =pd.read_csv('50_Startups.csv') 
X = dataset.iloc[:,:-1]
Y = dataset.iloc[:,-1]
#encodage de la variable state
from sklearn.preprocessing import LabelEncoder , OneHotEncoder 
labelEncoder_X=LabelEncoder()
X.iloc[:,-1]=labelEncoder_X.fit_transform(X.iloc[:,-1])
onehotencoder=OneHotEncoder(categorical_features=[-1])
X=onehotencoder.fit_transform(X).toarray()
X=X[:,[1,2,3,4,5]]
#spliting train /test
from sklearn.model_selection import train_test_split
X_train , X_test , Y_train , Y_test =train_test_split(X,Y,test_size=1.0/5,random_state=0)
#fitting multiple linear regression to the trainin set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)
y_pred=regressor.predict(X_test)
MSE= metrics.mean_squared_error (y_pred,Y_test)
RMSE =MSE**0.5
print (MSE,RMSE)
Z=np.array([0,0,130000,140000,300000]).reshape(1,-1)
W=regressor.predict(Z)

regressor.intercept_
regressor.coef_
#BackWard 
import statsmodels.api as sm
X=np.append(arr=np.ones((50,1)).astype(int), values=X, axis=1)
X_opt=X[:,[0,1,2,3,4,5]]
regressor_ols=sm.OLS(endog=Y,exog=X_opt).fit()
regressor_ols.summary()
#elimination du variable X2 
X_opt=X[:,[0,1,3,4,5]]
regressor_ols=sm.OLS(endog=Y,exog=X_opt).fit()
regressor_ols.summary()
#elimination du variable X1 
X_opt=X[:,[0,3,4,5]]
regressor_ols=sm.OLS(endog=Y,exog=X_opt).fit()
regressor_ols.summary()
#elimination du variable X2 
X_opt=X[:,[0,3,5]]
regressor_ols=sm.OLS(endog=Y,exog=X_opt).fit()
regressor_ols.summary()
#elimination du variable X2 
X_opt=X[:,[0,3]]
regressor_ols=sm.OLS(endog=Y,exog=X_opt).fit()
regressor_ols.summary()
