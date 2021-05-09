# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 12:32:10 2021

@author: Linda
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler = pd.read_csv("veriler.csv")
print(veriler)


x = veriler.iloc[:,1:4].values #bağımsız değişkenler
y = veriler.iloc[:,4:].values  #bağımlı değiskenler

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.33, random_state =0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

X_train = sc.fit_transform(x_train) 
X_test = sc.transform(x_test) 

#lojistik regresyon

from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state =0)

logr.fit(X_train, y_train) 

y_pred = logr.predict(X_test) 
print(y_test)


#confusion matrix
            
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(" ")
print("Confusion Matrix")
print(cm)


   