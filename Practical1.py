# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 06:01:29 2022

@author: Abhay
"""

import pandas as pd

csvfile = pd.read_csv("C:\\Users\\ladae\\OneDrive\\Documents\\data.csv")

print("rows: ",csvfile.shape[0] ," columns: " , csvfile.shape[1])

print(csvfile.head())

print(csvfile.City) 

for i in range(len(csvfile.City)):
    print(csvfile.City[i])
    
print("=====")
print(csvfile.City[0])

print("=====")
print(csvfile.iloc[0])

print("=====")
print(csvfile.City.iloc[[1,2]])

print("=====")
print(csvfile.City.iloc[[1,3]])

print("=====")
print(csvfile.iloc[[-0,-1,-2]])

print("=====")
print(csvfile.where(csvfile["City"] == "Bardoli").dropna())

print("=====")
print(csvfile.where(csvfile["Age"].isna()))

print("=====")
print(csvfile.dropna())

print("=====")
x = csvfile["Purchased"].mode()
print(csvfile.fillna(x))

