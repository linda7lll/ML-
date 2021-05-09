# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 00:34:35 2021

@author: Linda
"""

#Rassal agaçlar: 
    #veriler birden fazla parça bölünüyor, her parçadan bir ağaç meydana geliyor ve sonra oratk bir karar alınıyor. 

    
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

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=23, criterion = 'gini') #default olarak gini'le gerçekleştiriyor

rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)    

print("RFC: ")
print(cm)
