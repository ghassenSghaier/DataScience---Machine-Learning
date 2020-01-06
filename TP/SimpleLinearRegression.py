import matplotlib.pyplot as plt
import pandas as pd
dataset=pd.read_csv(‘Salary_Data’)
X=dataset.iloc[:,:-1]
Y=dataset.iloc[:,-1]
plt.scatter(X,y,color='red')
plt.title('Salary vs Experience')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()
