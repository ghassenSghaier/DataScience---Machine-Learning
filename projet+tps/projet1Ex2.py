

#importing the libraires
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics 

#imprting the dataset
dataset =pd.read_csv('PINS.csv') 
X = dataset.iloc[:,:-1]
Y = dataset.iloc[:,-1]

from  sklearn.preprocessing import Imputer
imputer =  Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer.fit(X.iloc[:,1].values.reshape(-1,1))
X.iloc[:,1]=imputer.transform(X.iloc[:,1].values.reshape(-1,1))

#encodage de la variable state
from sklearn.preprocessing import LabelEncoder , OneHotEncoder 
labelEncoder_X=LabelEncoder()
X.iloc[:,:-1]=labelEncoder_X.fit_transform(X.iloc[:,:-1])
onehotencoder=OneHotEncoder(categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()

#X=X[:,[0,1,2]]


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

#Coefficient of Determination -- Adjusted R²

from sklearn.metrics import r2_score
r_squared = r2_score(Y_test,y_pred)
adjusted_r_squared = 1 - (1-r_squared)*(len(Y)-1)/(len(Y)-X.shape[1]-1)

#Coefficients of the MLR and Evaluation of the model


Constant=regressor.intercept_
Coefficients=regressor.coef_

from sklearn.metrics import mean_squared_error
MSE=mean_squared_error(y_pred, Y_test)
RMSE=np.sqrt(MSE)

from sklearn import metrics
MAE= metrics.mean_absolute_error(y_pred,Y_test)
MSE= metrics.mean_squared_error(y_pred,Y_test) 
RMSE= metrics.mean_squared_error(y_pred,Y_test)**0.5


#BackWard 
import statsmodels.api as sm
X=np.append(arr=np.ones((39,1)).astype(int), values=X, axis=1)
X_opt=X[:,[0,1,2]]
regressor_ols=sm.OLS(endog=Y,exog=X_opt).fit()
regressor_ols.summary()

#elimination du variable X2 
X_opt=X[:,[0,2]]
regressor_ols=sm.OLS(endog=Y,exog=X_opt).fit()
regressor_ols.summary()

#elimination du variable X2 
X_opt=X[:,[0,1]]
regressor_ols=sm.OLS(endog=Y,exog=X_opt).fit()
regressor_ols.summary()

#elimination du variable X2 
X_opt=X[:,[1,2]]
regressor_ols=sm.OLS(endog=Y,exog=X_opt).fit()
regressor_ols.summary()

Z=np.array([0,1,19.5]).reshape(1,-1)
W=regressor.predict(Z)

Z1=np.array([1,0,18.9]).reshape(1,-1)
W1=regressor.predict(Z1)















