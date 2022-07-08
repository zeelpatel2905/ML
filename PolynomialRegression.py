# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 11:50:30 2022

@author: Abhay
"""
# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Prepare dataset for polynomial regression
polyreg = PolynomialFeatures(degree=3)
x_poly = polyreg.fit_transform(x)
linreg = LinearRegression()
linreg.fit(x_poly, y)

# visualising the polynomial results
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color='red')
plt.plot(x_grid, linreg.predict(
    polyreg.fit_transform(x_grid)),
    color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.show()

print(linreg.predict(polyreg.fit_transform([[6.5]])))

