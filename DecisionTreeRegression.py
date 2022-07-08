# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 12:52:40 2022

@author: Abhay
"""

# Decision tree regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:,-1].values

# Training the decision tree regression model on the whole dataset
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(x,y)

print(regressor.predict([[6.5]]))

plt.scatter(x, y, color = 'red')
plt.plot(x, regressor.predict(x), color = 'blue')
plt.title('Decision tree regressor')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


