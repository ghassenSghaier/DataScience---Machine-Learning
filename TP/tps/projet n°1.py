# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 08:38:36 2019

@author: LENOVO
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics 
#importation de la dataset 
#dataset=pd.read_csv("PINS.csv",sep=';')
dataset=pd.read_csv("PINS.csv")
X = dataset.iloc[:,:-1]
Y = dataset.iloc[:,-1]

#remplacement des données manquantes
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(X.iloc[:,[-1]])
X.iloc[:,[-1]]=imputer.transform(X.iloc[:,[-1]])

X_Blanc=X.iloc[:21,1]
X_Jaune=X.iloc[21:,1]

Y_Blanc=Y.iloc[:21]
Y_Jaune=Y.iloc[21:]


plt.scatter(X_Blanc,Y_Blanc,color="red")
plt.title("hauteur en fonction de diametre")
plt.xlabel('diametre')
plt.ylabel('hauteur')
plt.show()

plt.scatter(X_Jaune,Y_Jaune,color="red")
plt.title("hauteur en fonction de diametre")
plt.xlabel('diametre')
plt.ylabel('hauteur')
plt.show()

sigma=np.corrcoef(X_Blanc,Y_Blanc)
r = (np.mean(X_Blanc*Y_Blanc)- np.mean(X_Blanc)* np.mean(Y_Blanc))/(np.std(X_Blanc)*np.std(Y_Blanc))


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
X_Blanc_1 = X_Blanc.values.reshape(-1,1)
regressor.fit(X_Blanc_1,Y_Blanc)
y_pred=regressor.predict(X_Blanc_1)
plt.scatter(X_Blanc_1,Y_Blanc,color='red')
plt.plot(X_Blanc_1,y_pred,color='blue')

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
X_Jaune_1 = X_Jaune.values.reshape(-1,1)
regressor.fit(X_Jaune_1,Y_Jaune)
y_pred_1=regressor.predict(X_Jaune_1)
plt.scatter(X_Jaune_1,Y_Jaune,color='red')
plt.plot(X_Jaune_1,y_pred_1,color='blue')

from sklearn.preprocessing import LabelEncoder , OneHotEncoder 
labelEncoder_X=LabelEncoder()
X.iloc[:,0]=labelEncoder_X.fit_transform(X.iloc[:,0])
onehotencoder=OneHotEncoder(categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()
X=X[:,[1,2]]
a=regressor.intercept_
b=regressor.coef_
print (a,b)

