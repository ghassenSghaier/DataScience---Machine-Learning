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

#Fitting Decision Tree regression to the dataset
from sklearn.tree import DecisionTreeRegressor 
regressor2=DecisionTreeRegressor(criterion="mse")
regressor2.fit(X,Y)

Z_pred=regressor2.predict(X)
Y_pred=regressor2.predict(6.5)

plt.scatter(X,Y,Color= 'red')
plt.plot(X,regressor2.predict(X),color='blue')
plt.title('Truth or bluff (Decision Tree regression)')
plt.xlabel('Position label')
plt.ylabel('Salary')
plt.show()

X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,Y,Color= 'magenta')
plt.plot(X_grid,regressor2.predict(X_grid),color='green')
plt.title('Truth or bluff (Decision Tree Regression')
plt.xlabel('Position label')
plt.ylabel('Salary')
plt.show()


X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,Y,color='red')
plt.plot(X_grid,regressor2.predict(X_grid),color='blue')
plt.title('Truth or bluff (Decision Tree Regression')
plt.xlabel('Position label')
plt.ylabel('Salary')
plt.show()


#creterion = MSE
#Fitting Decision Tree regression to the dataset
from sklearn.tree import DecisionTreeRegressor 
regressor2=DecisionTreeRegressor(criterion="mse")
regressor2.fit(X,Y)
X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,Y,color='red')
plt.plot(X_grid,regressor2.predict(X_grid),color='blue')

from sklearn.tree import DecisionTreeRegressor 
regressor2=DecisionTreeRegressor(criterion="mse")
regressor2.fit(X,Y)
X_grid=np.arange(min(X),max(X),0.01)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,Y,color='red')
plt.plot(X_grid,regressor2.predict(X_grid),color='blue')



