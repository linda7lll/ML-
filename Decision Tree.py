# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 00:22:15 2021

@author: Linda
"""

#decision tree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#veriler oku
veriler = pd.read_csv("veriler.csv")
print(veriler)

#veri çek
x = veriler.iloc[:,1:4].values
y = veriler.iloc[:,4:].values

#test, train olarak böl
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33, random_state =0)

#ölçekleme
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier() #default olarak gini'le gerçekleştiriyor

dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)    

print("DTC: ")
print(cm)

