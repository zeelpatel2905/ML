# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 10:56:13 2022

@author: Abhay
"""
# KNN Classifier

# Impoerting the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the training set and the test set
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25, random_state=0)

# Feature scaling 
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Training the logistic regression model on the training set
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(x_train, y_train)

# Predecting the test set results
y_pred = classifier.predict(x_test)

# Making the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# visualizing the training set results
x_set, y_set = x_train, y_train
x1, x2 = np.meshgrid(np.arange(start = x_set[:,0].min() - 1,
                               stop = x_set[:,0].max() + 1, step = 0.01),
                    np.arange(start = x_set[:,1].min() -1, 
                               stop = x_set[:1].max() + 1, step = 0.01))

plt.contourf(x1,x2,classifier.predict(
    np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
    alpha = 0.75, cmap = ListedColormap(('orange', 'yellow')))

plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j,1],
                c = ['red', 'green'][i], label = j)
plt.title('KNN (training set)')
plt.xlabel('Age')
plt.ylabel('Estimated salary')
plt.legend()
plt.show()

# visualizing te test set results
x_set, y_set = x_test, y_test
x1, x2 = np.meshgrid(np.arange(start = x_set[:,0].min() - 1,
                               stop = x_set[:,0].max() + 1, step = 0.01),
                    np.arange(start = x_set[:,1].min() -1, 
                               stop = x_set[:1].max() + 1, step = 0.01))

plt.contourf(x1,x2,classifier.predict(
    np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
    alpha = 0.75, cmap = ListedColormap(('orange', 'yellow')))

plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j,1],
                c = ['red', 'green'][i], label = j)
plt.title('KNN (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated salary')
plt.legend()
plt.show()

