#region importing the librairies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#end region

# import the dataset
dataset = pd.read_csv('PINS.csv');
dataset.shape
dataset.describe()

Y1=dataset.loc[dataset['Varietes'] == 'blanc',['Hauteur']]
X1=dataset.loc[dataset['Varietes'] == 'blanc',['Diameteres']]
Y2=dataset.loc[dataset['Varietes'] == 'jaune',['Hauteur']]
X2=dataset.loc[dataset['Varietes'] == 'jaune',['Diameteres']]

 #integer-location based indexing / selection
#X1= dataset.iloc[0:21,1].values.reshape(-1,1)
#Y1= dataset.iloc[0:21,2].values.reshape(-1,1)
#X2= dataset.iloc[21:39,1].values.reshape(-1,1)
#Y2= dataset.iloc[21:39,2].values.reshape(-1,1)

#X = dataset['MinTemp'].values.reshape(-1,1)
#y = dataset['MaxTemp'].values.reshape(-1,1)

from  sklearn.preprocessing import Imputer
imputer =  Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer.fit(X1)
X1=imputer.transform(X1)
imputer.fit(X2)
X2=imputer.transform(X2)

sigma=np.corrcoef(X1,Y1)
r = (np.mean(X1*Y1)- np.mean(X1)* np.mean(Y1))/(np.std(X1)*np.std(Y1))

sigma1=np.corrcoef(X2,Y2)
r1 = (np.mean(X2*Y2)- np.mean(X2)* np.mean(Y2))/(np.std(X2)*np.std(Y2))


plt.scatter(X1,Y1,color='black')
plt.title('Hauteur vs Diamètre')
plt.xlabel('Diamètre')
plt.ylabel('Hauteur')
plt.show()


plt.scatter(X2,Y2,color='yellow')
plt.title('Hauteur vs Diamètre')
plt.xlabel('Diamètre')
plt.ylabel('Hauteur')
plt.show()

#Varietes Blanc

from sklearn.model_selection import train_test_split
X1_train,X1_test,Y1_train,Y1_test =train_test_split(X1,Y1,test_size =0.3, random_state= 0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X1_train=sc.fit_transform(X1_train)
X1_test=sc.fit_transform(X1_test)

plt.scatter(X1_train,Y1_train,color='red')
plt.title('Hauteur vs Diamètre')
plt.xlabel('Diamètre')
plt.ylabel('Hauteur')
plt.show()

from sklearn.linear_model import LinearRegression
from sklearn import metrics
regressor=LinearRegression()
regressor.fit(X1_train,Y1_train)

#y = b +mx
#To retrieve the intercept: (b)
print(regressor.intercept_)
#For retrieving the slope: (m)
print(regressor.coef_)

Y1_pred = regressor.predict(X1_test)

MAE = metrics.mean_absolute_error(Y1_pred,Y1_test)
MSE= metrics.mean_squared_error(Y1_pred,Y1_test )
RMSE= metrics.mean_absolute_error(Y1_pred,Y1_test )**0.5

from sklearn.metrics import r2_score
r2_score(Y1_test, Y1_pred)

plt.scatter(X1_train,Y1_train,color='black')
plt.plot(X1_train,regressor.predict(X1_train),color='blue')
#plt.plot(X1_test, Y1_pred, color='red', linewidth=2)
plt.title('diamètre vs hauteur')
plt.xlabel('Diamètre')
plt.ylabel('Hauteur')
plt.show()

#Varietes Jaune

from sklearn.model_selection import train_test_split
X2_train,X2_test,Y2_train,Y2_test =train_test_split(X2,Y2,test_size =0.3, random_state= 0)


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X2_train=sc.fit_transform(X2_train)
X2_test=sc.fit_transform(X2_test)

plt.scatter(X2_train,Y2_train,color='purple')
plt.title('Hauteur vs Diamètre')
plt.xlabel('Diamètre')
plt.ylabel('Hauteur')
plt.show()

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X2_train,Y2_train)
Y2_pred=regressor.predict(X2_test)

#y = b +mx
#To retrieve the intercept: (b)
print(regressor.intercept_)
#For retrieving the slope: (m)
print(regressor.coef_)
#df = pd.DataFrame({'Actual': Y1_test.flatten(), 'Predicted': Y1_pred.flatten()})

MAE = metrics.mean_absolute_error(Y2_pred,Y2_test)
MSE= metrics.mean_squared_error(Y2_pred,Y2_test )
RMSE= metrics.mean_absolute_error(Y2_pred,Y2_test )**0.5

from sklearn.metrics import r2_score
r2_score(Y2_test,Y2_pred)

plt.scatter(X2_train,Y2_train,color='yellow')
plt.plot(X2_train,regressor.predict(X2_train),color='blue')
plt.title('diamètre vs hauteur')
plt.xlabel('Diamètre')
plt.ylabel('Hauteur')
plt.show()