# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 17:05:29 2019

@author: luciano
"""

from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing

path = "D:/python/hackathon_carreiras/data/"
path_out = "D:/python/hackathon_carreiras/data/"

df_file = pd.read_csv(path + "bank-additional-full.csv",";")

#Remove atributo não disponivel para previsão
df_file = df_file.drop(["duration"],axis=1)

describe = df_file.describe()

corr = df_file.corr()

#split df nominal e numerico e label
df_obj = df_file.select_dtypes(include=[object])
columns_obj = df_obj.columns
df_nominal = df_obj.drop(["y"],axis=1)
df_numerico = df_file.drop(columns_obj,axis=1)
columns_obj = columns_obj.drop('y')
df_label = df_obj.drop(columns_obj,axis=1)

#one hot encode atributos nominais
df_nominal = pd.get_dummies(df_nominal)



#Removendo outliers
df_file['col'] = df_file['col'].clip(0,1000)

#scaling
for col in columsList:             
    max_c = df_c[col].max()
    df_c[col] = df_c.apply(lambda x: 0 if x[col]==0 else x[col]/max_c  , axis = 1)        
return df_c
   

df_final = pd.merge(df_nominal,df_numerico,left_index=True, right_index=True)  
df_final = pd.merge(df_final,df_label,left_index=True, right_index=True)  