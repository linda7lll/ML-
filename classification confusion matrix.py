# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 03:56:44 2021

@author: Linda
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics as mt

import scikitplot as skplt

irisVeriler = pd.read_csv('iris.csv')

print(irisVeriler)

x = irisVeriler.iloc[:,1:4].values 
y = irisVeriler.iloc[:,4:].values 
print(y)

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)



from sklearn.metrics import confusion_matrix

#%%

#k-NN

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski')

knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test,y_pred)

#metriklerin özeti
# mtt = mt.classification_report(y_test, y_pred)
# print(mtt)
skpltConf=skplt.plot_confusion_matrix(y_test, y_pred)
print(skpltConf)

print(" ")
print("KNN - Confusion Matrix :")
print(cm)
print("  ")
#%%

#support vector machine(svm)

from sklearn.svm import SVC
svc = SVC(kernel='rbf')

svc.fit(X_train,y_train)
y_pred = svc.predict(X_test)



cm = confusion_matrix(y_test,y_pred)

print("Support Vector Machine(SVM) - Confusion Matrix :")
print(cm)
print(" ")


#%%

#decision tree
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = 'entropy')

dtc.fit(X_train,y_train)
y_pred = dtc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)

print("Decision Tree - Confusion Matrix : ")
print(cm)
print("  ")


#%%





