# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 12:51:17 2022

@author: ladae
"""

# Importing the libraries
import numpy as np
import  matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("Market_Basket_Optimisation.csv", header=None)
transactions = []
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])    

# Traing apriori on the dataset 
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# Visualising the results
results = list(rules)