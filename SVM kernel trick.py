# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 17:29:07 2021

@author: Linda
"""

#SVM Kernel Trick/ Çekirdek hilesi
   #doğrusal olmayan, non-linear verileri ayrıştırma 
   #RBF
   

  


#import
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


from sklearn.svm import SVC
svc = SVC(kernel="rbf") #değişik kernellara göre başarı oranı artabilir
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)    

print("SVC: ")
print(cm)



