import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
dataset=pd.read_csv("Position_Salaries.csv")
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