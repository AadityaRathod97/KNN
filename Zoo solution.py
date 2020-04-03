# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 23:15:49 2020

@author: DELL
"""

'''
Question 1
Implement a KNN model to classify the animals in to categorie
'''


# Importing Libraries 
import pandas as pd
import numpy as np


animals = pd.read_csv("Zoo.csv")
animals.type.value_counts()

#coverting the string values into numeric 
animals.drop(["animal name"],axis=1,inplace=True)

# Training and Test data using 
from sklearn.model_selection import train_test_split
train,test = train_test_split(animals,test_size = 0.2) # 0.2 => 20 percent of entire data 

# KNN using sklearn 
# Importing Knn algorithm from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier as KNC

# for 3 nearest neighbours 
neigh = KNC(n_neighbors= 3)

# Fitting with training data 
neigh.fit(train.iloc[:,0:16],train.iloc[:,16])
# train accuracy 
train_acc = np.mean(neigh.predict(train.iloc[:,0:16])==train.iloc[:,16]) # 94 %

# test accuracy
test_acc = np.mean(neigh.predict(test.iloc[:,0:16])==test.iloc[:,16]) # 100%
test["pred"] = neigh.predict(test.iloc[:,0:16])
# creating empty list variable 
acc = []

# running KNN algorithm for 3 to 50 nearest neighbours(odd numbers) and 
# storing the accuracy values 
 
for i in range(3,50,2):
    neigh = KNC(n_neighbors=i)
    neigh.fit(train.iloc[:,0:16],train.iloc[:,16])
    train_acc = np.mean(neigh.predict(train.iloc[:,0:16])==train.iloc[:,16])
    test_acc = np.mean(neigh.predict(test.iloc[:,0:16])==test.iloc[:,16])
    acc.append([train_acc,test_acc])

acc


# k = 3 is giving the best acuracy
import matplotlib.pyplot as plt # library to do visualizations 

# train accuracy plot 
plt.plot(np.arange(3,50,2),[i[0] for i in acc],"bo-")
# test accuracy plot
plt.plot(np.arange(3,50,2),[i[1] for i in acc],"ro-");plt.legend(["train","test"])
