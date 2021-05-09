# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 13:28:11 2021

@author: Linda
"""

#kutuphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%

#eksik veriler/missing values

veriler = pd.read_csv("eksikveriler.csv")
print(veriler)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan,strategy="mean")

yas = veriler.iloc[:,1:4].values
print(yas)

imputer = imputer.fit(yas[:,1:4])
yas[:,1:4] = imputer.transform(yas[:,1:4])
print(yas)

#%%
from sklearn.impute import SimpleImputer
imputer =SimpleImputer(missing_values =np.nan, strategy="mean") 

veriler= pd.read_csv("eksikveriler.csv") 
print(veriler)

yas = veriler.iloc[:,1:4].values 
print(yas)

imputer = imputer.fit(yas[:,1:4]) 

yas[:,1:4]=imputer.transform(yas[:,1:4]) 

print(yas)

#%%

#kategorik -> sayısal
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
 
ulke[:,0] = le.fit_transform(veriler.iloc[:,0])
print(ulke)

ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)

#%%

#train, test, dataframe
sonuc = pd.DataFrame(data= ulke, index = range(22), columns = ["fr", "tr","us"])
print(sonuc)

yas = veriler.iloc[:,1:4].values
print(yas)

sonuc2 = pd.DataFrame(data = yas, index = range(22), columns = ["boy","kilo","yas"])
print(sonuc2)

cinsiyet = veriler.iloc[:,-1].values
print(cinsiyet)

sonuc3 = pd.DataFrame(data=cinsiyet, index = range(22), columns = ["cinsiyet"])
print(sonuc3)

#concat ile birleştirecegiz

s = pd.concat([sonuc,sonuc2],axis = 1)
print(s)

s1 = pd.concat([s,sonuc3],axis = 1)
print(s1)

from sklearn.model_selection import train_test_split                            
x_train, x_test, y_train, y_test = train_test_split(s,sonuc3,test_size=0.33, random_state =0)
                                       
#%%

#öznitelik ölçekleme

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
Y_train = sc.fit_transform(x_test)
Y_test = sc.fit_transform(y_test)


#%%