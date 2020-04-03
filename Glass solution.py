# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 23:18:20 2020

@author: DELL
"""

'''
Question 2
Prepare a model for glass classification using KNN
'''


# Importing Libraries 
import pandas as pd
import numpy as np

glass = pd.read_csv("glass.csv")

# Training and Test data using 
from sklearn.model_selection import train_test_split
train_glass,test_glass = train_test_split(glass,test_size = 0.2) # 0.2 => 20 percent of entire data 

# KNN using sklearn 
# Importing Knn algorithm from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier as KNC

# for 3 nearest neighbours 
neigh_glass = KNC(n_neighbors= 3)
neigh_glass.fit(glass.iloc[:,0:9],glass.iloc[:,9])

train_glass_acc = np.mean(neigh_glass.predict(train_glass.iloc[:,0:9]) == train_glass.iloc[:,9])

test_glass_acc = np.mean(neigh_glass.predict(test_glass.iloc[:,0:9]) == test_glass.iloc[:,9])

glass_pred = []

for i in range(3,50,2):
    neigh_glass = KNC(n_neighbors= i)
    neigh_glass.fit(glass.iloc[:,0:9],glass.iloc[:,9])
    train_glass_acc = np.mean(neigh_glass.predict(train_glass.iloc[:,0:9]) == train_glass.iloc[:,9])
    test_glass_acc = np.mean(neigh_glass.predict(test_glass.iloc[:,0:9]) == test_glass.iloc[:,9])
    glass_pred.append([train_glass_acc,test_glass_acc])
    

# k = 3 is giving the best acuracy
import matplotlib.pyplot as plt # library to do visualizations 

# train accuracy plot 
plt.plot(np.arange(3,50,2),[i[0] for i in glass_pred],"bo-")
# test accuracy plot
plt.plot(np.arange(3,50,2),[i[1] for i in glass_pred],"ro-");plt.legend(["train","test"])
